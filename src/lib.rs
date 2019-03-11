//! General implementation of the ES strategy described in https://arxiv.org/pdf/1703.03864.pdf

#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;
extern crate rand;
extern crate rayon;

use std::cmp::Ordering;
use std::io::prelude::*;
use std::fs::File;
use rand::distributions::Normal;
use rand::prelude::*;
use rayon::prelude::*;

pub type Float = f32;
#[cfg(feature = "floats-f64")]
pub type Float = f64;

//TODO:
//AdamaxBound ?


/// Definition of standard evaluator trait.
pub trait Evaluator
{
    /// Function to evaluate a set of parameters given as parameter.
    /// Return the score towards the target (optimizer maximizes).
    /// Only used once per optimization call (only for the returned score).
    fn eval_test(&self, &[Float]) -> Float;
    /// Function to evaluate a set of parameters also given the loop index as parameter.rand
    /// In addition to the parameters, the loop index is provided to allow selection of the same batch.
    /// Return the score towards the target (optimizer maximizes).
    /// Only used during training (very often).
    fn eval_train(&self, &[Float], usize) -> Float;
}

/// Definition of the optimizer traits, to dynamically allow different optimizers
pub trait Optimizer
{
    /// Function to compute the delta step/update later applied to the parameters
    /// Takes parameters and gradient as input
    /// Returns delta vector
    fn get_delta(&mut self, &[Float], &[Float]) -> Vec<Float>;
}

/// SGD Optimizer, which actually is SGA here (stochastic gradient ascent)
/// Momentum and weight decay is available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGD
{
    lr:Float, //learning rate
    lambda:Float, //weight decay coefficient
    beta:Float, //momentum coefficient
    lastv:Vec<Float>, //last momentum gradient
}

impl SGD
{
    /// Create new SGD optimizer instance using default hyperparameters (lr = 0.01)
    pub fn new() -> SGD
    {
        SGD { lr: 0.01, lambda: 0.0, beta: 0.0, lastv: vec![0.0] }
    }
    
    /// Set learning rate
    pub fn set_lr(&mut self, learning_rate:Float) -> &mut Self
    {
        if learning_rate <= 0.0
        {
            panic!("Learning rate must be greater than zero!");
        }
        self.lr = learning_rate;
        
        self
    }
    
    /// Set lambda factor for weight decay
    pub fn set_lambda(&mut self, coeff:Float) -> &mut Self
    {
        if coeff < 0.0
        {
            panic!("Lambda coefficient may not be smaller than zero!");
        }
        self.lambda = coeff;
        
        self
    }
    
    /// Set beta factor for momentum
    pub fn set_beta(&mut self, factor:Float) -> &mut Self
    {
        if factor < 0.0 || factor >= 1.0
        {
            panic!(format!("Prohibited momentum paramter: {}. Must be in [0.0, 1.0)!", factor));
        }
        self.beta = factor;
        
        self
    }
    
    /// Encodes the optimizer as a JSON string.
    pub fn to_json(&self) -> String
    {
        serde_json::to_string(self).expect("Encoding JSON failed!")
    }

    /// Builds a new optimizer from a JSON string.
    pub fn from_json(encoded:&str) -> SGD
    {
        serde_json::from_str(encoded).expect("Decoding JSON failed!")
    }
    
    /// Saves the model to a file
    pub fn save(&self, file:&str) -> Result<(), std::io::Error>
    {
        let mut file = File::create(file)?;
        let json = self.to_json();
        file.write_all(json.as_bytes())?;
        Ok(())
    }
    
    /// Creates a model from a previously saved file
    pub fn load(file:&str) -> Result<SGD, std::io::Error>
    {
        let mut file = File::open(file)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        Ok(SGD::from_json(&json))
    }
}

impl Optimizer for SGD
{
    /// Compute delta update from params and gradient
    fn get_delta(&mut self, params:&[Float], grad:&[Float]) -> Vec<Float>
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

/// Adam Optimizer, with possibility of using AdaBound
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam
{
    lr:Float, //learning rate
    lambda:Float, //weight decay coefficient
    beta1:Float, //exponential moving average factor
    beta2:Float, //exponential second moment average factor (squared gradient)
    eps:Float, //small epsilon to avoid divide by zero (fuzz factor)
    t:usize, //number of taken timesteps
    avggrad1:Vec<Float>, //first order moment (avg)
    avggrad2:Vec<Float>, //second oder moment (squared)
    adabound:bool, //switch whether to use the AdaBound variant
    final_lr:Float, //final LR to use using AdaBound (SGD)
    gamma:Float, //convergence speed of bounding functions for AdaBound
}

impl Adam
{
    /// Create new Adam optimizer instance using default hyperparameters (lr = 0.001, lambda = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
    /// adabound = false, final_lr = 0.1, gamma: 0.001)
    /// Also try higher LR; beta2 = 0.99; try adabound!
    pub fn new() -> Adam
    {
        Adam { lr: 0.001, lambda: 0.0, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, avggrad1: vec![0.0], avggrad2: vec![0.0],
            adabound: false, final_lr: 0.1, gamma: 0.001 }
    }
    
    /// Set learning rate
    pub fn set_lr(&mut self, learning_rate:Float) -> &mut Self
    {
        if learning_rate <= 0.0
        {
            panic!("Learning rate must be greater than zero!");
        }
        self.lr = learning_rate;
        
        self
    }
    
    /// Set final learning rate for AdaBound (SGD)
    pub fn set_final_lr(&mut self, learning_rate:Float) -> &mut Self
    {
        if learning_rate <= 0.0
        {
            panic!("Learning rate must be greater than zero!");
        }
        self.final_lr = learning_rate;
        
        self
    }
    
    /// Set lambda factor for weight decay
    pub fn set_lambda(&mut self, coeff:Float) -> &mut Self
    {
        if coeff < 0.0
        {
            panic!("Lambda coefficient may not be smaller than zero!");
        }
        self.lambda = coeff;
        
        self
    }
    
    /// Set gamma factor for AdaBound bounding convergence
    pub fn set_gamma(&mut self, coeff:Float) -> &mut Self
    {
        if coeff < 0.0 || coeff >= 1.0
        {
            panic!("Gamma coefficient is in appropriate!");
        }
        self.gamma = coeff;
        
        self
    }
    
    /// Set beta1 coefficient (for exponential moving average of first moment)
    pub fn set_beta1(&mut self, beta:Float) -> &mut Self
    {
        if beta < 0.0 || beta >= 1.0
        {
            panic!(format!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta));
        }
        self.beta1 = beta;
        
        self
    }
    
    /// Set beta2 coefficient (for exponential moving average of second moment)
    pub fn set_beta2(&mut self, beta:Float) -> &mut Self
    {
        if beta < 0.0 || beta >= 1.0
        {
            panic!(format!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta));
        }
        self.beta2 = beta;
        
        self
    }
    
    /// Set epsilon to avoid divide by zero (fuzz factor)
    pub fn set_eps(&mut self, epsilon:Float) -> &mut Self
    {
        if epsilon < 0.0
        {
            panic!("Epsilon must be >= 0!");
        }
        self.eps = epsilon;
        
        self
    }
    
    /// Set usage of AdaBound
    pub fn set_adabound(&mut self, use_bound:bool) -> &mut Self
    {
        self.adabound = use_bound;
        self
    }
    
    /// Retrieve the timestep (to allow computing manual learning rate decay)
    pub fn get_t(&self) -> usize
    {
        self.t
    }
    
    /// Encodes the optimizer as a JSON string.
    pub fn to_json(&self) -> String
    {
        serde_json::to_string(self).expect("Encoding JSON failed!")
    }

    /// Builds a new optimizer from a JSON string.
    pub fn from_json(encoded:&str) -> Adam
    {
        serde_json::from_str(encoded).expect("Decoding JSON failed!")
    }
    
    /// Saves the model to a file
    pub fn save(&self, file:&str) -> Result<(), std::io::Error>
    {
        let mut file = File::create(file)?;
        let json = self.to_json();
        file.write_all(json.as_bytes())?;
        Ok(())
    }
    
    /// Creates a model from a previously saved file
    pub fn load(file:&str) -> Result<Adam, std::io::Error>
    {
        let mut file = File::open(file)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        Ok(Adam::from_json(&json))
    }
}

impl Optimizer for Adam
{
    /// Compute delta update from params and gradient
    fn get_delta(&mut self, params:&[Float], grad:&[Float]) -> Vec<Float>
    {
        if self.avggrad1.len() != params.len() || self.avggrad2.len() != params.len()
        { //initialize with zero moments
            self.avggrad1 = vec![0.0; params.len()];
            self.avggrad2 = vec![0.0; params.len()];
        }
        
        //timestep + unbias factor
        self.t += 1;
        let lr_unbias = self.lr * (1.0 - self.beta2.powf(self.t as Float)).sqrt() / (1.0 - self.beta1.powf(self.t as Float));
        //dynamic bound
        let lower_bound = (1.0 - 1.0 / (self.gamma * self.t as Float + 1.0)) * self.final_lr;
        let upper_bound = (1.0 + 1.0 / (self.gamma * self.t as Float)) * self.final_lr;
        
        //update exponential moving averages and compute delta (parameter update)
        let mut delta = grad.to_vec();
        for (((g1, g2), d), p) in self.avggrad1.iter_mut().zip(self.avggrad2.iter_mut()).zip(delta.iter_mut()).zip(params.iter())
        {
            //moment 1 and 2 update
            *g1 = self.beta1 * *g1 + (1.0 - self.beta1) * *d;
            *g2 = self.beta2 * *g2 + (1.0 - self.beta2) * *d * *d;
            //delta update
            if self.adabound
            {
                //dynamic bound
                let bound_lr = (lr_unbias / (g2.sqrt() + self.eps)).max(lower_bound).min(upper_bound);
                *d = bound_lr * *g1;
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adamax
{
    lr:Float, //learning rate
    lambda:Float, //weight decay coefficient
    beta1:Float, //exponential moving average factor
    beta2:Float, //exponential second moment average factor (squared gradient)
    eps:Float, //small epsilon to avoid divide by zero (fuzz factor)
    t:usize, //number of taken timesteps
    avggrad1:Vec<Float>, //first order moment (avg)
    avggrad2:Vec<Float>, //second oder moment (squared)
}

impl Adamax
{
    /// Create new Adamax optimizer instance using default hyperparameters (lr = 0.002, lambda = 0, beta1 = 0.9, beta2 = 0.999, eps = 0)
    /// Also try higher LR; beta2 = 0.99
    pub fn new() -> Adamax
    {
        Adamax { lr: 0.002, lambda: 0.0, beta1: 0.9, beta2: 0.999, eps: 0.0, t: 0, avggrad1: vec![0.0], avggrad2: vec![0.0] }
    }
    
    /// Set learning rate
    pub fn set_lr(&mut self, learning_rate:Float) -> &mut Self
    {
        if learning_rate <= 0.0
        {
            panic!("Learning rate must be greater than zero!");
        }
        self.lr = learning_rate;
        
        self
    }
    
    /// Set lambda factor for weight decay
    pub fn set_lambda(&mut self, coeff:Float) -> &mut Self
    {
        if coeff < 0.0
        {
            panic!("Lambda coefficient may not be smaller than zero!");
        }
        self.lambda = coeff;
        
        self
    }
    
    /// Set beta1 coefficient (for exponential moving average of first moment)
    pub fn set_beta1(&mut self, beta:Float) -> &mut Self
    {
        if beta < 0.0 || beta >= 1.0
        {
            panic!(format!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta));
        }
        self.beta1 = beta;
        
        self
    }
    
    /// Set beta2 coefficient (for exponential moving average of second moment)
    pub fn set_beta2(&mut self, beta:Float) -> &mut Self
    {
        if beta < 0.0 || beta >= 1.0
        {
            panic!(format!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta));
        }
        self.beta2 = beta;
        
        self
    }
    
    /// Set epsilon to avoid divide by zero (fuzz factor)
    pub fn set_eps(&mut self, epsilon:Float) -> &mut Self
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
    
    /// Encodes the optimizer as a JSON string.
    pub fn to_json(&self) -> String
    {
        serde_json::to_string(self).expect("Encoding JSON failed!")
    }

    /// Builds a new optimizer from a JSON string.
    pub fn from_json(encoded:&str) -> Adamax
    {
        serde_json::from_str(encoded).expect("Decoding JSON failed!")
    }
    
    /// Saves the model to a file
    pub fn save(&self, file:&str) -> Result<(), std::io::Error>
    {
        let mut file = File::create(file)?;
        let json = self.to_json();
        file.write_all(json.as_bytes())?;
        Ok(())
    }
    
    /// Creates a model from a previously saved file
    pub fn load(file:&str) -> Result<Adamax, std::io::Error>
    {
        let mut file = File::open(file)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        Ok(Adamax::from_json(&json))
    }
}

impl Optimizer for Adamax
{
    /// Compute delta update from params and gradient
    fn get_delta(&mut self, params:&[Float], grad:&[Float]) -> Vec<Float>
    {
        if self.avggrad1.len() != params.len() || self.avggrad2.len() != params.len()
        { //initialize with zero moments
            self.avggrad1 = vec![0.0; params.len()];
            self.avggrad2 = vec![0.0; params.len()];
        }
        
        //timestep + unbias factor
        self.t += 1;
        let lr_unbias = self.lr / (1.0 - self.beta1.powf(self.t as Float));
        
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
    params:Vec<Float>, //current parameters
    opt:Opt, //chosen optimizer
    eval:Feval, //evaluator function
    
    std:Float, //standard deviation to calculate the noise for parameters
    samples:usize, //number of mirror-samples per step to approximate the gradient
}

impl<Feval:Evaluator> ES<Feval, SGD>
{
    /// Shortcut for ES::new(...) using SGD:
    /// Create a new ES-Optimizer using SGA (create SGD object with the given parameters).
    pub fn new_with_sgd(evaluator:Feval, learning_rate:Float, beta:Float, lambda:Float) -> ES<Feval, SGD>
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
    /// Shortcut for ES::new(...) using Adam (/AdaBound):
    /// Create a new ES-Optimizer using Adam (create Adam object with the given parameters, rest left to default).
    /// Change these paramters using method get_opt_mut().set_<...>(...).
    pub fn new_with_adam(evaluator:Feval, learning_rate:Float, lambda:Float) -> ES<Feval, Adam>
    {
        let mut optimizer = Adam::new();
        optimizer.set_lr(learning_rate)
            .set_lambda(lambda);
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
    
    /// Shortcut for ES::new(...) using Adam (/AdaBound):
    /// Create a new ES-Optimizer using Adam (create Adam object with the given parameters).
    pub fn new_with_adam_ex(evaluator:Feval, learning_rate:Float, lambda:Float, beta1:Float, beta2:Float, adabound:bool, final_lr:Float) -> ES<Feval, Adam>
    {
        let mut optimizer = Adam::new();
        optimizer.set_lr(learning_rate)
            .set_lambda(lambda)
            .set_beta1(beta1)
            .set_beta2(beta2)
            .set_adabound(adabound)
            .set_final_lr(final_lr);
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
}

impl<Feval:Evaluator> ES<Feval, Adamax>
{
    /// Shortcut for ES::new(...) using Adamax:
    /// Create a new ES-Optimizer using Adamax (create Adam object with the given parameters, rest left to default).
    /// Change these paramters using method get_opt_mut().set_<...>(...).
    pub fn new_with_adamax(evaluator:Feval, learning_rate:Float, lambda:Float) -> ES<Feval, Adamax>
    {
        let mut optimizer = Adamax::new();
        optimizer.set_lr(learning_rate)
            .set_lambda(lambda);
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
    }
    
    /// Shortcut for ES::new(...) using Adam:
    /// Create a new ES-Optimizer using Adam (create Adam object with the given parameters).
    pub fn new_with_adamax_ex(evaluator:Feval, learning_rate:Float, lambda:Float, beta1:Float, beta2:Float, eps:Float) -> ES<Feval, Adamax>
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
    pub fn set_params(&mut self, params:Vec<Float>) -> &mut Self
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
    pub fn set_std(&mut self, noise:Float) -> &mut Self
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
    pub fn get_params(&self) -> &Vec<Float>
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
    pub fn get_params_mut(&mut self) -> &mut Vec<Float>
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
    pub fn optimize(&mut self, n:usize) -> (Float, Float)
    {
        let seed = random::<u64>() % (std::u64::MAX - self.samples as u64);
        
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            grad = vec![0.0; self.dim];
            for i in 0..self.samples
            {
                //repeatable eps generation to save memory
                let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                //generate random epsilon
                let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
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
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
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
    pub fn optimize_ranked(&mut self, n:usize) -> (Float, Float)
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
                    let centered_rank = rank as Float / (self.samples as Float - 0.5) - 1.0;
                    for (g, e) in grad.iter_mut().zip(eps.iter())
                    {
                        *g += *e * negfactor * centered_rank;
                    }
                });
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval_test(&self.params), norm(&grad))
    }
    
    /// Optimize for n steps.
    /// Uses the standardized scores to calculate the gradients.
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize_std(&mut self, n:usize) -> (Float, Float)
    {
        let seed = random::<u64>() % (std::u64::MAX - self.samples as u64);
        
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            grad = vec![0.0; self.dim];
            //first generate and fill whole vector of scores
            let mut scores = vec![(0.0, 0.0); self.samples];
            scores.iter_mut().enumerate().for_each(|(i, (scorepos, scoreneg))|
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
            //calculate std, mean
            let (_mean, std) = get_mean_std(&mut scores);
            //sum up and calculate gradient from the sum
            scores.iter().enumerate().for_each(|(i, (scorepos, scoreneg))|
                {
                    let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                    let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    for (g, e) in grad.iter_mut().zip(eps.iter())
                    {
                        //subtraction by mean cancels out
                        *g += *e * (*scorepos - *scoreneg) / std;
                    }
                });
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval_test(&self.params), norm(&grad))
    }
    
    /// Optimize for n steps.
    /// Uses the normalized scores to calculate the gradients.
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize_norm(&mut self, n:usize) -> (Float, Float)
    {
        let seed = random::<u64>() % (std::u64::MAX - self.samples as u64);
        
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            grad = vec![0.0; self.dim];
            //first generate and fill whole vector of scores
            let mut scores = vec![(0.0, 0.0); self.samples];
            let mut maximum = -1.0;
            scores.iter_mut().enumerate().for_each(|(i, (scorepos, scoreneg))|
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
                    //calculate maxmimum absolute score
                    if scorepos.abs() > maximum
                    {
                        maximum = scorepos.abs();
                    }
                    if scoreneg.abs() > maximum
                    {
                        maximum = scoreneg.abs();
                    }
                });
            //sum up and calculate gradient from the sum
            scores.iter().enumerate().for_each(|(i, (scorepos, scoreneg))|
                {
                    let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                    let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    for (g, e) in grad.iter_mut().zip(eps.iter())
                    {
                        //subtraction by mean cancels out
                        *g += *e * (*scorepos - *scoreneg) / maximum;
                    }
                });
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
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
    pub fn optimize_par(&mut self, n:usize) -> (Float, Float)
        where Opt:Sync, Feval:Sync
    {
        let seed = random::<u64>() % (std::u64::MAX - self.samples as u64);
        
        let mut grad = vec![0.0; self.dim];
        //for n iterations:
        for _i in 0..n
        {
            //approximate gradient with self.samples double-sided samples
            grad = (0..self.samples).into_par_iter().map(|i|
                {
                    //repeatable eps generation to save memory
                    let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                    //gen and compute test parameters
                    let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    let mut testparampos = eps.clone();
                    let mut testparamneg = eps.clone();
                    for ((pos, neg), p) in testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
                    {
                        *pos = *p + *pos;
                        *neg = *p - *neg;
                    }
                    //evaluate parameters to compute scores
                    let scorepos = self.eval.eval_train(&testparampos, _i);
                    let scoreneg = self.eval.eval_train(&testparamneg, _i);
                    //compute gradient parts and the sum up in reduce to calculate the gradient
                    mul_scalar(&mut eps, scorepos - scoreneg);
                    eps
                }).reduce(|| vec![0.0; self.dim], |mut a, b| { add_inplace(&mut a, &b); a });
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
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
    pub fn optimize_ranked_par(&mut self, n:usize) -> (Float, Float)
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
                    let centered_rank = rank as Float / (self.samples as Float - 0.5) - 1.0;
                    mul_scalar(&mut eps, negfactor * centered_rank);
                    eps
                }).reduce(|| vec![0.0; self.dim], |mut a, b| { add_inplace(&mut a, &b); a });
                //if reduce saves too much and takes too much memory: do serial (normal iter) and initialize grad before,
                //sum components to grad in loop (for_each);
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval_test(&self.params), norm(&grad))
    }
    
    /// Optimize for n steps (in parallel).
    /// Optimizer and Evaluator must satisfy the Sync trait.
    /// Uses the standardized scores to calculate the gradients.
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize_std_par(&mut self, n:usize) -> (Float, Float)
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
            //calculate std, mean
            let (_mean, std) = get_mean_std(&mut scores);
            //sum up and calculate gradient from the sum
            grad = scores.par_iter().enumerate().map(|(i, (scorepos, scoreneg))|
                {
                    let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                    let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    //subtraction by mean cancels out
                    mul_scalar(&mut eps, (*scorepos - *scoreneg) / std);
                    eps
                }).reduce(|| vec![0.0; self.dim], |mut a, b| { add_inplace(&mut a, &b); a });
                //if reduce saves too much and takes too much memory: do serial (normal iter) and initialize grad before,
                //sum components to grad in loop (for_each);
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        (self.eval.eval_test(&self.params), norm(&grad))
    }
    
    /// Optimize for n steps (in parallel).
    /// Optimizer and Evaluator must satisfy the Sync trait.
    /// Uses the normalized scores to calculate the gradients.
    /// Returns a tuple (score, gradnorm), which is the latest parameters' evaluated score and the norm of the last gradient/delta change.
    pub fn optimize_norm_par(&mut self, n:usize) -> (Float, Float)
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
            //calculate maxmimum absolute score
            let mut maximum = -1.0;
            scores.iter().for_each(|x|
                {
                    if x.0.abs() > maximum
                    {
                        maximum = x.0.abs();
                    }
                    if x.1.abs() > maximum
                    {
                        maximum = x.1.abs();
                    }
                });
            //sum up and calculate gradient from the sum
            grad = scores.par_iter().enumerate().map(|(i, (scorepos, scoreneg))|
                {
                    let mut rng = SmallRng::seed_from_u64(seed + i as u64);
                    let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
                    //subtraction by mean cancels out
                    mul_scalar(&mut eps, (*scorepos - *scoreneg) / maximum);
                    eps
                }).reduce(|| vec![0.0; self.dim], |mut a, b| { add_inplace(&mut a, &b); a });
                //if reduce saves too much and takes too much memory: do serial (normal iter) and initialize grad before,
                //sum components to grad in loop (for_each);
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
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
fn gen_rnd_vec_rng<RNG: Rng>(rng:&mut RNG, n:usize, std:Float) -> Vec<Float>
{
    let normal = Normal::new(0.0, std as f64);
    normal.sample_iter(rng).take(n).map(|x| x as Float).collect()
}

/// Generate a vector of random numbers with 0 mean and std std, normally distributed.
/// Using standard thread_rng.
pub fn gen_rnd_vec(n:usize, std:Float) -> Vec<Float>
{
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, std as f64);
    normal.sample_iter(&mut rng).take(n).map(|x| x as Float).collect()
}

/// Add a second vector onto the first vector in place
fn add_inplace(v1:&mut [Float], v2:&[Float])
{
    for (val1, val2) in v1.iter_mut().zip(v2.iter())
    {
        *val1 += *val2;
    }
}

/// Multiplies a scalar to a vector
fn mul_scalar(vec:&mut [Float], scalar:Float)
{
    for val in vec.iter_mut()
    {
        *val *= scalar;
    }
}

/// Calculates the norm of a vector
fn norm(vec:&[Float]) -> Float
{
    let mut norm = 0.0;
    for val in vec.iter()
    {
        norm += *val * *val;
    }
    norm.sqrt()
}

/// calculate mean and standard deviation of the scores
fn get_mean_std(vec:&[(Float, Float)]) -> (Float, Float)
{
    let mut mean = 0.0;
    vec.iter().for_each(|(scorepos, scoreneg)| { mean += *scorepos + *scoreneg; });
    mean /= (2 * vec.len()) as Float;
    
    let mut std = 0.0;
    vec.iter().for_each(|(scorepos, scoreneg)|
        {
            let mut diff = *scorepos - mean;
            std += diff * diff;
            diff = *scoreneg - mean;
            std += diff * diff;
        });
    std /= (2 * vec.len()) as Float;
    std = std.sqrt();
    
    (mean, std)
}

/// Sorts the internal score-vector, so that ranks can be computed
fn sort_scores<T,U>(vec:&mut Vec<(T, U, Float)>)
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
