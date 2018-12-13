//! General implementation of the ES strategy described in https://arxiv.org/pdf/1703.03864.pdf

extern crate rand;
extern crate rayon;

use rand::distributions::{Normal, Distribution};
use rayon::prelude::*;

//TODO:


/// Definition of evaluator traits
pub trait Evaluator
{
    /// Function to evaluate a set of parameters given as parameter
    /// Return the score towards the target (optimizer maximizes)
    fn eval(&self, &Vec<f64>) -> f64;
}

/// Definition of the optimizer traits, to dynamically allow different optimizers
pub trait Optimizer
{
    /// Function to compute the delta step/update later applied to the parameters
    /// Takes parameters and gradient as input
    /// Returns delta vector
    fn get_delta(&mut self, &Vec<f64>, &Vec<f64>) -> Vec<f64>;
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
    fn get_delta(&mut self, params:&Vec<f64>, grad:&Vec<f64>) -> Vec<f64>
    {
        if self.lastv.len() != params.len()
        { //initialize with zero gradient
            self.lastv = vec![0.0; params.len()];
        }
        
        //calculate momentum update and compute delta (parameter update)
        let mut delta = grad.clone();
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
    fn get_delta(&mut self, params:&Vec<f64>, grad:&Vec<f64>) -> Vec<f64>
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
        let mut delta = grad.clone();
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
    fn get_delta(&mut self, params:&Vec<f64>, grad:&Vec<f64>) -> Vec<f64>
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
        let mut delta = grad.clone();
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
#[derive(Clone)]
pub struct ES<Feval:Evaluator+Clone, Opt:Optimizer+Clone>
{
    dim:usize, //problem dimensionality
    params:Vec<f64>, //current parameters
    opt:Opt, //chosen optimizer
    eval:Feval, //evaluator function
    
    std:f64, //standard deviation to calculate the noise for parameters
    samples:usize, //number of mirror-samples per step to approximate the gradient
}

impl<Feval:Evaluator+Clone> ES<Feval, SGD>
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

impl<Feval:Evaluator+Clone> ES<Feval, Adam>
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

impl<Feval:Evaluator+Clone> ES<Feval, Adamax>
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
    pub fn new_with_adam_ex(evaluator:Feval, learning_rate:f64, lambda:f64, beta1:f64, beta2:f64, eps:f64) -> ES<Feval, Adamax>
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

impl<Feval:Evaluator+Clone, Opt:Optimizer+Clone> ES<Feval, Opt>
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
    
    /// Optimize for n steps
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
                //generate epsilon
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
                let scorepos = self.eval.eval(&testparampos);
                let scoreneg = self.eval.eval(&testparamneg);
                //calculate grad sum update
                for (g, e) in grad.iter_mut().zip(eps.iter())
                {
                    *g += *e * scorepos - *e * scoreneg;
                }
            }
            //calculate gradient from the sum
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as f64 * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval(&self.params), norm(&grad))
    }
    
    /// Optimize for n steps (evaluation in parallel)
    /// Optimizer and Evaluator must satisfy the Sync trait
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize_par(&mut self, n:usize) -> (f64, f64)
        where Opt:Sync, Feval:Sync
    {
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            //first generate a set of eps vectors
            let mut epsvec = Vec::new();
            for _ in 0..self.samples
            {
                let mut eps = gen_rnd_vec(self.dim, self.std);
                epsvec.push(eps.clone());
                mul_scalar(&mut eps, -1.0);
                epsvec.push(eps);
            }
            //then evaluate them in parallel
            epsvec.par_iter_mut().for_each(|eps|
                {
                    let mut testparam = eps.clone();
                    add_inplace(&mut testparam, &self.params);
                    let score = self.eval.eval(&testparam);
                    mul_scalar(eps, score);
                });
            //afterwards add up the results to compute the gradient
            grad = epsvec.pop().expect("Number of sample was zero - impossible!");
            while !epsvec.is_empty()
            {
                let res = epsvec.pop().expect("Epsvec.is_empty did not work!");
                add_inplace(&mut grad, &res);
            }
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as f64 * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval(&self.params), norm(&grad))
    }
}

/// Generate a vector of random numbers with 0 mean and std std, normally distributed.
pub fn gen_rnd_vec(n:usize, std:f64) -> Vec<f64>
{
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, std);
    normal.sample_iter(&mut rng).take(n).collect()
}

/// Add a second vector onto the first vector in place
fn add_inplace(v1:&mut Vec<f64>, v2:&Vec<f64>)
{
    for (val1, val2) in v1.iter_mut().zip(v2.iter())
    {
        *val1 += *val2;
    }
}

/// Multiplies a scalar to a vector
fn mul_scalar(vec:&mut Vec<f64>, scalar:f64)
{
    for val in vec.iter_mut()
    {
        *val *= scalar;
    }
}

/// Calculates the norm of a vector
fn norm(vec:&Vec<f64>) -> f64
{
    let mut norm = 0.0;
    for val in vec.iter()
    {
        norm += *val * *val;
    }
    norm.sqrt()
}
