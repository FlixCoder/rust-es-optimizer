//! General implementation of the ES strategy described in https://arxiv.org/pdf/1703.03864.pdf

extern crate rand;
extern crate rayon;

use std::cmp::Ordering;
use rand::distributions::Normal;
use rand::prelude::*;
use rayon::prelude::*;

//TODO:
//implement score standardization additionnaly/in either of the methods


/// Definition of standard evaluator trait.
pub trait Evaluator
{
    /// Function to evaluate a set of parameters given as parameter.
    /// Return the score towards the target (optimizer maximizes).
    /// Only used once per optimization call (only for the returned score).
    fn eval_test(&self, &[f64]) -> f64;
    /// Function to evaluate a set of parameters also given the loop index as parameter.rand
    /// In addition to the parameters, the loop index is provided to allow selection of the same batch.
    /// Return the score towards the target (optimizer maximizes).
    /// Only used during training (very often).
    fn eval_train(&self, &[f64], usize) -> f64;
}

/// Definition of the optimizer traits, to dynamically allow different optimizers
pub trait Optimizer
{
    /// Function to compute the delta step/update later applied to the parameters
    /// Takes parameters and gradient as input
    /// Returns delta vector
    fn get_delta(&mut self, &[f64], &[f64]) -> Vec<f64>;
}

/// SGD Optimizer, which actually is SGA here (stochastic gradient ascent)
/// Momentum and weight decay is available
#[derive(Debug, Clone)]
pub struct SGD
{
    lr:f64, //learning rate
    lambda:f64, //weight decay coefficient
    beta:f64, //momentum coefficient
    lastv:Vec<f64>, //last momentum gradient
}

impl SGD
{
    /// Create new SGD optimizer instance using default hyperparameters (lr = 0.01)
    pub fn new() -> SGD
    {
        SGD { lr: 0.01, lambda: 0.0, beta: 0.0, lastv: vec![0.0] }
    }
    
    pub fn set_lr(&mut self, learning_rate:f64) -> &mut Self
    {
        if learning_rate <= 0.0
        {
            panic!("Learning rate must be greater than zero!");
        }
        self.lr = learning_rate;
        
        self
    }
    
    /// Set lambda factor for weight decay
    pub fn set_lambda(&mut self, coeff:f64) -> &mut Self
    {
        if coeff < 0.0
        {
            panic!("Lambda coefficient may not be smaller than zero!");
        }
        self.lambda = coeff;
        
        self
    }
    
    /// Set beta factor for momentum
    pub fn set_beta(&mut self, factor:f64) -> &mut Self
    {
        if factor < 0.0 || factor >= 1.0
        {
            panic!(format!("Prohibited momentum paramter: {}. Must be in [0.0, 1.0)!", factor));
        }
        self.beta = factor;
        
        self
    }
}

impl Optimizer for SGD
{
    /// Compute delta update from params and gradient
    fn get_delta(&mut self, params:&[f64], grad:&[f64]) -> Vec<f64>
    {
        if self.lastv.len() != params.len()
        { //initialize with zero gradient
            self.lastv = vec![0.0; params.len()];
        }
        
        //calculate momentum update and compute delta (parameter update)
        let mut delta = grad.to_vec();
        for ((m, d), p) in self.lastv.iter_mut().zip(delta.iter_mut()).zip(params.iter())
        {
            //momentum update
            *m = self.beta * *m + (1.0 - self.beta) * *d;
            //compute delta based on momentum
            *d = self.lr * *m; //here no minus, because ascend instead of descent
            //add weight decay
            *d -= self.lr * self.lambda * *p;
        }
        
        //return
        delta
    }
}

/// Adam Optimizer
#[derive(Debug, Clone)]
pub struct Adam
{
    lr:f64, //learning rate
    lambda:f64, //weight decay coefficient
    beta1:f64, //exponential moving average factor
    beta2:f64, //exponential second moment average factor (squared gradient)
    eps:f64, //small epsilon to avoid divide by zero (fuzz factor)
    t:usize, //number of taken timesteps
    avggrad1:Vec<f64>, //first order moment (avg)
    avggrad2:Vec<f64>, //second oder moment (squared)
    grad2max:Vec<f64>, //maximum avggrad2 vector, to implement amsgrad
    amsgrad:bool, //indicate wether to use amsgrad
}

impl Adam
{
    /// Create new Adam optimizer instance using default hyperparameters (lr = 0.001, lambda = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, amsgrad = false)
    /// Also try higher LR; beta2 = 0.99
    pub fn new() -> Adam
    {
        Adam { lr: 0.001, lambda: 0.0, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, avggrad1: vec![0.0], avggrad2: vec![0.0], grad2max: vec![0.0], amsgrad: false }
    }
    
    pub fn set_lr(&mut self, learning_rate:f64) -> &mut Self
    {
        if learning_rate <= 0.0
        {
            panic!("Learning rate must be greater than zero!");
        }
        self.lr = learning_rate;
        
        self
    }
    
    /// Set lambda factor for weight decay
    pub fn set_lambda(&mut self, coeff:f64) -> &mut Self
    {
        if coeff < 0.0
        {
            panic!("Lambda coefficient may not be smaller than zero!");
        }
        self.lambda = coeff;
        
        self
    }
    
    /// Set beta1 coefficient (for exponential moving average of first moment)
    pub fn set_beta1(&mut self, beta:f64) -> &mut Self
    {
        if beta < 0.0 || beta >= 1.0
        {
            panic!(format!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta));
        }
        self.beta1 = beta;
        
        self
    }
    
    /// Set beta2 coefficient (for exponential moving average of second moment)
    pub fn set_beta2(&mut self, beta:f64) -> &mut Self
    {
        if beta < 0.0 || beta >= 1.0
        {
            panic!(format!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta));
        }
        self.beta2 = beta;
        
        self
    }
    
    /// Set epsilon to avoid divide by zero (fuzz factor)
    pub fn set_eps(&mut self, epsilon:f64) -> &mut Self
    {
        if epsilon < 0.0
        {
            panic!("Epsilon must be >= 0!");
        }
        self.eps = epsilon;
        
        self
    }
    
    /// Set whether to use AMSGrad or not
    pub fn set_amsgrad(&mut self, amsgrad:bool) -> &mut Self
    {
        self.amsgrad = amsgrad;
        
        self
    }
    
    /// Retrieve the timestep (to allow computing manual learning rate decay)
    pub fn get_t(&self) -> usize
    {
        self.t
    }
}

impl Optimizer for Adam
{
    /// Compute delta update from params and gradient
    fn get_delta(&mut self, params:&[f64], grad:&[f64]) -> Vec<f64>
    {
        if self.avggrad1.len() != params.len() || self.avggrad2.len() != params.len()
        { //initialize with zero moments
            self.avggrad1 = vec![0.0; params.len()];
            self.avggrad2 = vec![0.0; params.len()];
            if self.amsgrad
            {
                self.grad2max = vec![0.0; params.len()];
            }
        }
        
        //timestep + unbias factor
        self.t += 1;
        let lr_unbias = self.lr * (1.0 - self.beta2.powf(self.t as f64)).sqrt() / (1.0 - self.beta1.powf(self.t as f64));
        
        //update exponential moving averages and compute delta (parameter update)
        let mut delta = grad.to_vec();
        for (i, (((g1, g2), d), p)) in self.avggrad1.iter_mut().zip(self.avggrad2.iter_mut()).zip(delta.iter_mut()).zip(params.iter()).enumerate()
        {
            //moment 1 and 2 update
            *g1 = self.beta1 * *g1 + (1.0 - self.beta1) * *d;
            *g2 = self.beta2 * *g2 + (1.0 - self.beta2) * *d * *d;
            //delta update
            if self.amsgrad
            {
                self.grad2max[i] = self.grad2max[i].max(*g2); //amsgrad update
                *d = lr_unbias * *g1 / (self.grad2max[i].sqrt() + self.eps); //normally it would be -lr_unbias, but we want to maximize
            }
            else
            {
                *d = lr_unbias * *g1 / (g2.sqrt() + self.eps); //normally it would be -lr_unbias, but we want to maximize
            }
            //weight decay
            *d -= self.lr * self.lambda * *p;
        }
        
        //return
        delta
    }
}

/// Adam Optimizer
#[derive(Debug, Clone)]
pub struct Adamax
{
    lr:f64, //learning rate
    lambda:f64, //weight decay coefficient
    beta1:f64, //exponential moving average factor
    beta2:f64, //exponential second moment average factor (squared gradient)
    eps:f64, //small epsilon to avoid divide by zero (fuzz factor)
    t:usize, //number of taken timesteps
    avggrad1:Vec<f64>, //first order moment (avg)
    avggrad2:Vec<f64>, //second oder moment (squared)
}

impl Adamax
{
    /// Create new Adamax optimizer instance using default hyperparameters (lr = 0.002, lambda = 0, beta1 = 0.9, beta2 = 0.999, eps = 0)
    /// Also try higher LR; beta2 = 0.99
    pub fn new() -> Adamax
    {
        Adamax { lr: 0.002, lambda: 0.0, beta1: 0.9, beta2: 0.999, eps: 0.0, t: 0, avggrad1: vec![0.0], avggrad2: vec![0.0] }
    }
    
    pub fn set_lr(&mut self, learning_rate:f64) -> &mut Self
    {
        if learning_rate <= 0.0
        {
            panic!("Learning rate must be greater than zero!");
        }
        self.lr = learning_rate;
        
        self
    }
    
    /// Set lambda factor for weight decay
    pub fn set_lambda(&mut self, coeff:f64) -> &mut Self
    {
        if coeff < 0.0
        {
            panic!("Lambda coefficient may not be smaller than zero!");
        }
        self.lambda = coeff;
        
        self
    }
    
    /// Set beta1 coefficient (for exponential moving average of first moment)
    pub fn set_beta1(&mut self, beta:f64) -> &mut Self
    {
        if beta < 0.0 || beta >= 1.0
        {
            panic!(format!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta));
        }
        self.beta1 = beta;
        
        self
    }
    
    /// Set beta2 coefficient (for exponential moving average of second moment)
    pub fn set_beta2(&mut self, beta:f64) -> &mut Self
    {
        if beta < 0.0 || beta >= 1.0
        {
            panic!(format!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta));
        }
        self.beta2 = beta;
        
        self
    }
    
    /// Set epsilon to avoid divide by zero (fuzz factor)
    pub fn set_eps(&mut self, epsilon:f64) -> &mut Self
    {
        if epsilon < 0.0
        {
            panic!("Epsilon must be >= 0!");
        }
        self.eps = epsilon;
        
        self
    }
    
    /// Retrieve the timestep (to allow computing manual learning rate decay)
    pub fn get_t(&self) -> usize
    {
        self.t
    }
}

impl Optimizer for Adamax
{
    /// Compute delta update from params and gradient
    fn get_delta(&mut self, params:&[f64], grad:&[f64]) -> Vec<f64>
    {
        if self.avggrad1.len() != params.len() || self.avggrad2.len() != params.len()
        { //initialize with zero moments
            self.avggrad1 = vec![0.0; params.len()];
            self.avggrad2 = vec![0.0; params.len()];
        }
        
        //timestep + unbias factor
        self.t += 1;
        let lr_unbias = self.lr / (1.0 - self.beta1.powf(self.t as f64));
        
        //update exponential moving averages and compute delta (parameter update)
        let mut delta = grad.to_vec();
        for (((g1, g2), d), p) in self.avggrad1.iter_mut().zip(self.avggrad2.iter_mut()).zip(delta.iter_mut()).zip(params.iter())
        {
            //moment 1 and 2 update
            *g1 = self.beta1 * *g1 + (1.0 - self.beta1) * *d;
            *g2 = (self.beta2 * *g2).max(d.abs());
            //delta update
            *d = lr_unbias * *g1 / (*g2 + self.eps); //normally it would be -lr_unbias, but we want to maximize
            //weight decay
            *d -= self.lr * self.lambda * *p;
        }
        
        //return
        delta
    }
}


/// Evolution-Strategy optimizer class. Optimizes given parameters towards a maximum evaluation-score.
pub struct ES<Feval:Evaluator, Opt:Optimizer>
{
    dim:usize, //problem dimensionality
    params:Vec<f64>, //current parameters
    opt:Opt, //chosen optimizer
    eval:Feval, //evaluator function
    
    std:f64, //standard deviation to calculate the noise for parameters
    samples:usize, //number of mirror-samples per step to approximate the gradient
}

impl<Feval:Evaluator> ES<Feval, SGD>
{
    /// Shortcut for ES::new(...) using SGD:
    /// Create a new ES-Optimizer using SGA (create SGD object with the given parameters).
    pub fn new_with_sgd(evaluator:Feval, learning_rate:f64, beta:f64, lambda:f64) -> ES<Feval, SGD>
    {
        let mut optimizer = SGD::new();
        optimizer.set_lr(learning_rate)
            .set_beta(beta)
            .set_lambda(lambda);
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
}

impl<Feval:Evaluator> ES<Feval, Adam>
{
    /// Shortcut for ES::new(...) using Adam:
    /// Create a new ES-Optimizer using Adam (create Adam object with the given parameters, rest left to default).
    /// Change these paramters using method get_opt_mut().set_<...>(...).
    pub fn new_with_adam(evaluator:Feval, learning_rate:f64, lambda:f64) -> ES<Feval, Adam>
    {
        let mut optimizer = Adam::new();
        optimizer.set_lr(learning_rate)
            .set_lambda(lambda);
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
    
    /// Shortcut for ES::new(...) using Adam:
    /// Create a new ES-Optimizer using Adam (create Adam object with the given parameters).
    pub fn new_with_adam_ex(evaluator:Feval, learning_rate:f64, lambda:f64, beta1:f64, beta2:f64, eps:f64, amsgrad:bool) -> ES<Feval, Adam>
    {
        let mut optimizer = Adam::new();
        optimizer.set_lr(learning_rate)
            .set_lambda(lambda)
            .set_beta1(beta1)
            .set_beta2(beta2)
            .set_eps(eps)
            .set_amsgrad(amsgrad);
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
}

impl<Feval:Evaluator> ES<Feval, Adamax>
{
    /// Shortcut for ES::new(...) using Adamax:
    /// Create a new ES-Optimizer using Adamax (create Adam object with the given parameters, rest left to default).
    /// Change these paramters using method get_opt_mut().set_<...>(...).
    pub fn new_with_adamax(evaluator:Feval, learning_rate:f64, lambda:f64) -> ES<Feval, Adamax>
    {
        let mut optimizer = Adamax::new();
        optimizer.set_lr(learning_rate)
            .set_lambda(lambda);
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
    
    /// Shortcut for ES::new(...) using Adam:
    /// Create a new ES-Optimizer using Adam (create Adam object with the given parameters).
    pub fn new_with_adamax_ex(evaluator:Feval, learning_rate:f64, lambda:f64, beta1:f64, beta2:f64, eps:f64) -> ES<Feval, Adamax>
    {
        let mut optimizer = Adamax::new();
        optimizer.set_lr(learning_rate)
            .set_lambda(lambda)
            .set_beta1(beta1)
            .set_beta2(beta2)
            .set_eps(eps);
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
}

impl<Feval:Evaluator, Opt:Optimizer> ES<Feval, Opt>
{
    /// Create a new ES-Optimizer
    /// evaluator = object with Evaluator trait that computes the objetive-score based on the paramters
    /// optimizer = optimizer to calculate the parameter update using the gradient and the current parameters. (e.g. use SGD::new() aka SGA)
    /// Important: set the initial parameters afterswards by calling set_params to specify the problem dimension. (Default is [0.0], dim=1)
    pub fn new(optimizer:Opt, evaluator:Feval) -> ES<Feval, Opt>
    {
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
    
    /// Set the parameters (potentially reinitializing the process)
    /// params = set of parameters to optimize
    pub fn set_params(&mut self, params:Vec<f64>) -> &mut Self
    {
        self.params = params;
        self.dim = self.params.len();
        
        self
    }
    
    /// Change the optimizer
    pub fn set_opt(&mut self, optimizer:Opt) -> &mut Self
    {
        self.opt = optimizer;
        
        self
    }
    
    /// Change the evaluator function
    pub fn set_eval(&mut self, evaluator:Feval) -> &mut Self
    {
        self.eval = evaluator;
        
        self
    }
    
    /// Set noise's standard deviation (applied to the parameters)
    /// Humanoid example in the paper used 0.02 as an example (default).
    /// Probably best to choose in dependence of evaluator output size.
    /// Tweak learning rate to fit to the std.
    pub fn set_std(&mut self, noise:f64) -> &mut Self
    {
        if noise <= 0.0
        {
            panic!("Noise std may not be <= 0!");
        }
        self.std = noise;
        
        self
    }
    
    /// Set the number of mirror-samples per step to approximate the gradient
    /// Was probably around 700 in paper (1400 workers)
    pub fn set_samples(&mut self, num:usize) -> &mut Self
    {
        if num == 0
        {
            panic!("Number of samples cannot be zero!");
        }
        self.samples = num;
        
        self
    }
    
    /// Get the current parameters (as ref)
    pub fn get_params(&self) -> &Vec<f64>
    {
        &self.params
    }
    
    /// Get the optimizer (as ref)
    pub fn get_opt(&self) -> &Opt
    {
        &self.opt
    }
    
    /// Get the evaluator (as ref)
    pub fn get_eval(&self) -> &Feval
    {
        &self.eval
    }
    
    /// Get the current parameters (as mut)
    pub fn get_params_mut(&mut self) -> &mut Vec<f64>
    {
        &mut self.params
    }
    
    /// Get the optimizer (as mut, to change parameters)
    pub fn get_opt_mut(&mut self) -> &mut Opt
    {
        &mut self.opt
    }
    
    /// Get the evaluator (as mut, to change parameters)
    pub fn get_eval_mut(&mut self) -> &mut Feval
    {
        &mut self.eval
    }
    
    /// Optimize for n steps.
    /// Uses the evaluator's score to calculate the gradients.
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize(&mut self, n:usize) -> (f64, f64)
    {
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            grad = vec![0.0; self.dim];
            for _ in 0..self.samples
            {
                //generate random epsilon
                let eps = gen_rnd_vec(self.dim, self.std);
                //compute test parameters in both directions
                let mut testparampos = eps.clone();
                let mut testparamneg = eps.clone();
                for ((pos, neg), p) in testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
                {
                    *pos = *p + *pos;
                    *neg = *p - *neg;
                }
                //evaluate test parameters
                let scorepos = self.eval.eval_train(&testparampos, _i);
                let scoreneg = self.eval.eval_train(&testparamneg, _i);
                //calculate grad sum update
                for (g, e) in grad.iter_mut().zip(eps.iter())
                {
                    *g += *e * (scorepos - scoreneg);
                }
            }
            //calculate gradient from the sum
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as f64 * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval_test(&self.params), norm(&grad))
    }
    
    /// Optimize for n steps.
    /// Uses the centered ranks to calculate the gradients.
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize_ranked(&mut self, n:usize) -> (f64, f64)
    {
        let seed = random::<u64>() % (std::u64::MAX - self.samples as u64);
        
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            grad = vec![0.0; self.dim];
            //first generate and fill whole vector of scores
            let mut scores = Vec::new();
            for i in 0..self.samples
            {
                //repeatable eps generation to save memory
                let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                //gen and compute test parameters
                let mut testparampos = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
                let mut testparamneg = testparampos.clone();
                for ((pos, neg), p) in testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
                {
                    *pos = *p + *pos;
                    *neg = *p - *neg;
                }
                //evaluate parameters and save scores
                let scorepos = self.eval.eval_train(&testparampos, _i);
                let scoreneg = self.eval.eval_train(&testparamneg, _i);
                scores.push((i, false, scorepos));
                scores.push((i, true, scoreneg));
            }
            //sort, create ranks, sum up and calculate gradient from the sum
            sort_scores(&mut scores);
            scores.iter().enumerate().for_each(|(rank, (i, neg, _score))|
                {
                    let mut rng = SmallRng::seed_from_u64(seed + *i as u64);
                    let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    let negfactor = if *neg { -1.0 } else { 1.0 };
                    let centered_rank = rank as f64 / (self.samples as f64 - 0.5) - 1.0;
                    for (g, e) in grad.iter_mut().zip(eps.iter())
                    {
                        *g += *e * negfactor * centered_rank;
                    }
                });
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as f64 * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval_test(&self.params), norm(&grad))
    }
    
    /// Optimize for n steps (evaluation in parallel).
    /// Uses the evaluator's score to calculate the gradients.
    /// Optimizer and Evaluator must satisfy the Sync trait.
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize_par(&mut self, n:usize) -> (f64, f64)
        where Opt:Sync, Feval:Sync
    {
        let seed = random::<u64>() % (std::u64::MAX - self.samples as u64);
        
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            //first generate and fill whole vector of scores
            let mut scores = vec![(0.0, 0.0); self.samples];
            scores.par_iter_mut().enumerate().for_each(|(i, (scorepos, scoreneg))|
                {
                    //repeatable eps generation to save memory
                    let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                    //gen and compute test parameters
                    let mut testparampos = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
                    let mut testparamneg = testparampos.clone();
                    for ((pos, neg), p) in testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
                    {
                        *pos = *p + *pos;
                        *neg = *p - *neg;
                    }
                    //evaluate parameters and save scores
                    *scorepos = self.eval.eval_train(&testparampos, _i);
                    *scoreneg = self.eval.eval_train(&testparamneg, _i);
                });
            //then add up to compute the gradient sequentially (could only do parallel with mutex on grad)
            /*grad = vec![0.0; self.dim];
            scores.iter().enumerate().for_each(|(i, (scorepos, scoreneg))|
                {
                    let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                    let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    for (g, e) in grad.iter_mut().zip(eps.iter())
                    {
                        *g += *e * (*scorepos - *scoreneg);
                    }
                });*/
            grad = scores.par_iter().enumerate().map(|(i, (scorepos, scoreneg))|
                {
                    let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                    let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    mul_scalar(&mut eps, *scorepos - *scoreneg);
                    eps
                }).reduce(|| vec![0.0; self.dim], |mut a, b| { add_inplace(&mut a, &b); a });
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as f64 * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval_test(&self.params), norm(&grad))
    }
    
    /// Optimize for n steps (evaluation in parallel).
    /// Uses the centered ranks to calculate the gradients.
    /// Optimizer and Evaluator must satisfy the Sync trait.
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize_ranked_par(&mut self, n:usize) -> (f64, f64)
        where Opt:Sync, Feval:Sync
    {
        let seed = random::<u64>() % (std::u64::MAX - self.samples as u64);
        
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            //first generate and fill whole vector of scores
            let mut scores = vec![(0, false, 0.0); 2*self.samples];
            for i in 0..self.samples
            {
                scores[2*i].0 = i;
                scores[2*i+1].0 = i;
                scores[2*i+1].1 = true;
            }
            scores.par_iter_mut().for_each(|(i, neg, score)|
                {
                    //repeatable eps generation to save memory
                    let mut rng = SmallRng::seed_from_u64(seed + *i as u64);
                    //gen and compute test parameters
                    let mut testparam = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
                    if *neg
                    {
                        mul_scalar(&mut testparam, -1.0);
                    }
                    add_inplace(&mut testparam, &self.params);
                    //evaluate parameters and save scores
                    *score = self.eval.eval_train(&testparam, _i);
                });
            //compute the centered ranks and calculate the summed result to compute the gradient
            sort_scores(&mut scores);
            grad = scores.par_iter().enumerate().map(|(rank, (i, neg, _score))|
                {
                    let mut rng = SmallRng::seed_from_u64(seed + *i as u64);
                    let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    let negfactor = if *neg { -1.0 } else { 1.0 };
                    let centered_rank = rank as f64 / (self.samples as f64 - 0.5) - 1.0;
                    mul_scalar(&mut eps, negfactor * centered_rank);
                    eps
                }).reduce(|| vec![0.0; self.dim], |mut a, b| { add_inplace(&mut a, &b); a });
                //if reduce saves too much and takes to much memory: do serial (normal iter) and initialize grad before,
                //sum components to grad in loop (for_each); see optimize_par for code
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as f64 * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval_test(&self.params), norm(&grad))
    }
}


/// Generate a vector of random numbers with 0 mean and std std, normally distributed.
/// Using specified RNG.
fn gen_rnd_vec_rng<RNG: Rng>(rng:&mut RNG, n:usize, std:f64) -> Vec<f64>
{
    let normal = Normal::new(0.0, std);
    normal.sample_iter(rng).take(n).collect()
}

/// Generate a vector of random numbers with 0 mean and std std, normally distributed.
/// Using standard thread_rng.
pub fn gen_rnd_vec(n:usize, std:f64) -> Vec<f64>
{
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, std);
    normal.sample_iter(&mut rng).take(n).collect()
}

/// Add a second vector onto the first vector in place
fn add_inplace(v1:&mut [f64], v2:&[f64])
{
    for (val1, val2) in v1.iter_mut().zip(v2.iter())
    {
        *val1 += *val2;
    }
}

/// Multiplies a scalar to a vector
fn mul_scalar(vec:&mut [f64], scalar:f64)
{
    for val in vec.iter_mut()
    {
        *val *= scalar;
    }
}

/// Calculates the norm of a vector
fn norm(vec:&[f64]) -> f64
{
    let mut norm = 0.0;
    for val in vec.iter()
    {
        norm += *val * *val;
    }
    norm.sqrt()
}

/// Sorts the internal score-vector, so that ranks can be computed
fn sort_scores<T,U>(vec:&mut Vec<(T, U, f64)>)
{ //worst score in front
    vec.sort_unstable_by(|ref r1, ref r2| { //partial cmp and check for NaN
            let r = (r1.2).partial_cmp(&r2.2);
            if r.is_some()
            {
                r.unwrap()
            }
            else
            {
                if r1.2.is_nan() { if r2.2.is_nan() { Ordering::Equal } else { Ordering::Less } } else { Ordering::Greater }
            }
        });
}
