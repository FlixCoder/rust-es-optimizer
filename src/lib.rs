//! General implementation of the ES strategy described in https://arxiv.org/pdf/1703.03864.pdf

extern crate rand;
extern crate rayon;

use rand::distributions::{Normal, Distribution};
use rayon::prelude::*;

//TODO:
//add Adam optimizer?
//adapt std to parameter/gradient values (std ~ param std)?
//adapt lr to gradient/parameter sizes?


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
        
        //momentum SGD update
        mul_scalar(&mut self.lastv, self.beta);
        let mut gradcopy = grad.clone();
        mul_scalar(&mut gradcopy, 1.0 - self.beta);
        add_inplace(&mut self.lastv, &gradcopy);
        gradcopy = self.lastv.clone();
        mul_scalar(&mut gradcopy, self.lr);
        
        //weight decay
        let mut wdecay = params.clone();
        mul_scalar(&mut wdecay, -self.lr * self.lambda);
        add_inplace(&mut gradcopy, &wdecay);
        
        //return
        gradcopy
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

impl<Feval:Evaluator+Clone, Opt:Optimizer+Clone> ES<Feval, Opt>
{
    /// Create a new ES-Optimizer
    /// params = set of parameters to optimize
    /// evaluator = function that computes the objetive-score based on the paramters
    /// default optimizer is SGD (which is actually SGA = stochastic gradient ascent)
    pub fn new(optimizer:Opt, evaluator:Feval) -> ES<Feval, Opt>
    {
        ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.05, samples: 500 }
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
    /// Humanoid example in the paper used 0.02 as an example.
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
                let mut testparampos = gen_rnd_vec(self.dim, self.std);
                let mut epspos = testparampos.clone();
                let mut testparamneg = testparampos.clone();
                mul_scalar(&mut testparamneg, -1.0);
                let mut epsneg = testparamneg.clone();
                add_inplace(&mut testparampos, &self.params);
                add_inplace(&mut testparamneg, &self.params);
                let score = self.eval.eval(&testparampos);
                mul_scalar(&mut epspos, score);
                add_inplace(&mut grad, &epspos);
                let score = self.eval.eval(&testparamneg);
                mul_scalar(&mut epsneg, score);
                add_inplace(&mut grad, &epsneg);
            }
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
        norm += *val * *val;
    }
    norm.sqrt()
}
