//! General implementation of the ES strategy described in https://arxiv.org/pdf/1703.03864.pdf

extern crate rand;

use rand::distributions::{Normal, Distribution};

//TODO:
//add Adam optimizer?


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
}

impl SGD
{
    /// Create new SGD optimizer instance using default hyperparameters (lr = 0.01)
    pub fn new() -> SGD
    {
        SGD { lr: 0.01, lambda: 0.0, beta: 0.0 }
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
        vec![0.0; 2]
    }
}


/// Evolution-Strategy optimizer class. Optimizes given parameters towards a maximum evaluation-score.
#[derive(Clone)]
pub struct ES<Feval, Opt:Optimizer>
    where Feval: Fn(&Vec<f64>) -> f64
{
    dim:usize, //problem dimensionality
    params:Vec<f64>, //current parameters
    opt:Opt, //chosen optimizer
    eval:Feval, //evaluator function
    
    std:f64, //standard deviation to calculate the noise for parameters
    samples:usize, //number of samples per step to approximate the gradient
}

impl<Feval, Opt:Optimizer> ES<Feval, Opt>
    where Feval: Fn(&Vec<f64>) -> f64
{
    /// Create a new ES-Optimizer
    /// params = set of parameters to optimize
    /// evaluator = function that computes the objetive-score based on the paramters
    /// default optimizer is SGD (which is actually SGA = stochastic gradient ascent)
    pub fn new(optimizer:Opt, params:Vec<f64>, evaluator:Feval) -> ES<Feval, Opt>
    {
        let dim = params.len();
        ES { dim: dim, params: params, opt: optimizer, eval: evaluator, std: 0.1, samples: 500 }
    }
    
    /// Change the optimizer
    pub fn set_opt(&mut self, optimizer:Opt) -> &mut Self
    {
        self.opt = optimizer;
        
        self
    }
    
    /// Set the parameters (potentially reinitializing the process)
    pub fn set_params(&mut self, params:Vec<f64>) -> &mut Self
    {
        self.params = params;
        self.dim = self.params.len();
        
        self
    }
    
    /// Change the evaluator function
    pub fn set_eval(&mut self, evaluator:Feval) -> &mut Self
    {
        self.eval = evaluator;
        
        self
    }
    
    /// Set noise's standard deviation (applied to the parameters)
    /// Humanoid example in the paper used 0.02 as an example
    pub fn set_std(&mut self, noise:f64) -> &mut Self
    {
        if noise <= 0.0
        {
            panic!("Noise std may not be <= 0!");
        }
        self.std = noise;
        
        self
    }
    
    /// Set the number of samples per step to approximate the gradient
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
    
    /// Get the optimizer (as ref, to change parameters)
    pub fn get_opt(&self) -> &Opt
    {
        &self.opt
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
            for _j in 0..self.samples
            {
                let mut eps = gen_rnd_vec(self.dim, self.std);
                let mut negeps = eps.clone();
                mul_scalar(&mut negeps, -1.0);
                add_inplace(&mut eps, &self.params);
                add_inplace(&mut negeps, &self.params);
                let score = (self.eval)(&eps);
                mul_scalar(&mut eps, score);
                add_inplace(&mut grad, &eps);
                let score = (self.eval)(&negeps);
                mul_scalar(&mut negeps, score);
                add_inplace(&mut grad, &negeps);
            }
            mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as f64 * self.std));
            //calculate the delta update using the optimizer
            let delta = self.opt.get_delta(&self.params, &grad);
            //update the parameters
            add_inplace(&mut self.params, &delta);
        }
        
        ((self.eval)(&self.params), norm(&grad))
    }
}

/// Generate a vector of random numbers with 0 mean and std std, normally distributed.
fn gen_rnd_vec(n:usize, std:f64) -> Vec<f64>
{
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, std);
    normal.sample_iter(&mut rng).take(n).collect()
}

/// Add a second vector onto the first vector in place
fn add_inplace(v1:&mut Vec<f64>, v2:&Vec<f64>)
{
    if v1.len() != v2.len()
    {
        panic!("Vectors are not equally sized!");
    }
    
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
        norm += *val;
    }
    norm / vec.len() as f64
}
