extern crate esopt;

use esopt::*;


fn main()
{
    //initialize required objects
    let mut opt = SGD::new(); //SGA optimizer (named SGD)
    opt.set_lr(0.5) //learning rate
        .set_beta(0.0) //momentum factor
        .set_lambda(0.0); //weight decay coefficient
    
    let eval = ExampleEval::new(25.0); //target = 25
    
    //create ES-Optimizer
    let mut es = ES::new(opt, eval); //Evolution-Strategy-Optimizer using optimizer and evaluator
    es.set_params(vec![0.0]) //initial parameters (important to specify the problem dimension, default is vec![0.0])
        .set_std(1.0) //parameter noise standard deviation to approximate the gradient
        .set_samples(50); //number of mirrored samples to use to approximate the gradient
    
    //track the optimizer's results
    for _ in 0..5
    {
        let res = es.optimize(15); //optimize for n steps
        println!("(Score, Gradnorm): {:?}", res);
        println!("Params: {:?}", es.get_params());
    }
}



//carrier object for evaluator, which can include training data or target information
#[derive(Clone)]
struct ExampleEval
{
    target:f64,
}

impl ExampleEval
{
    fn new(target:f64) -> ExampleEval
    {
        ExampleEval { target: target }
    }
}

//implement Evaluator trait to allow usage in the optimizer
impl Evaluator for ExampleEval
{
    //compute the negative absolute error (maximize to get close to target)
    fn eval(&self, params:&Vec<f64>) -> f64
    {
        let error = self.target - params[0];
        -error.abs()
    }
}
