# ES-Pptimizer [esopt]
General Evolution-Strategy-Optimizer implementation according to https://arxiv.org/abs/1703.03864 in Rust.

Simple example (see examples/simple.rs):
```rust
extern crate esopt;
use esopt::*;

fn main()
{
    //create required evaluator
    let eval = ExampleEval { target: 25.0 };
    
    //create Evolution-Strategy-Optimizer
    let mut es = ES::new_with_sgd(eval, 0.75, 0.0, 0.0); //using evaluator, lr, beta(momentum), lambda(weight decay)
    es.set_params(vec![0.0]) //initial parameters (important to specify the problem dimension, default is vec![0.0])
        .set_std(50.0) //parameter noise standard deviation to approximate the gradient
        .set_samples(50); //number of mirrored samples to use to approximate the gradient
    
    //track the optimizer's results
    for _ in 0..5
    {
        let res = es.optimize(2); //optimize for n steps
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
```
