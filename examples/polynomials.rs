extern crate esopt;

use esopt::*;

const DEGREE:usize = 2;
const REGULARIZE:f64 = 0.025; //own L1 reg. factor


fn main()
{
    //training data
    let points = vec![  (0.0, 0.1),
                        (1.0, 1.4),
                        (2.0, 4.0),
                        (3.0, 8.4),
                        (4.0, 16.5)
                    ];
    /*let points = vec![    (0.0, 0.0),
                        (1.0, 1.0),
                        (2.0, 4.0),
                        (3.0, 9.0),
                        (4.0, 16.0)
                    ];*/
    
    //initialize required objects
    let eval = PolynomeEval::new(points); //target
    let params = gen_rnd_vec(DEGREE+1, 0.1); //generate random initial parameters
    
    //create ES-Optimizer
    let mut es = ES::new_with_sgd(eval, 0.5, 0.1, 0.0); //Evolution-Strategy-Optimizer using optimizer and evaluator
    es.set_params(params) //initial parameters (important to specify the problem dimension, default is vec![0.0])
        .set_std(0.02) //parameter noise standard deviation to approximate the gradient
        .set_samples(50); //number of mirrored samples to use to approximate the gradient
    
    //track the optimizer's results
    for i in 0..5
    {
        let n = 20;
        let res = es.optimize(n); //optimize for n steps
        println!("After {:5} iterations:", (i+1) * n);
        println!("-MAE: {:7.4}", res.0);
        println!("Gradnorm: {:7.5}", res.1);
        print_f(es.get_params());
        println!("");
    }
}


fn calc_f(factors:&Vec<f64>, x:f64) -> f64
{
    let mut result = factors[0];
    let mut current_pow = 1.0;
    for i in 1..factors.len()
    {
        current_pow *= x;
        result += factors[i] * current_pow;
    }
    result
}

fn print_f(fct:&Vec<f64>)
{
    let mut str = String::new();
    for i in 0..fct.len()
    {
        str.push_str(&format!("{:7.4}x^{} + ", fct[i], i));
    }
    let tmp = str.len() - 3;
    str.truncate(tmp);
    println!("{}", str);
}


#[derive(Clone)]
struct PolynomeEval
{
    target:Vec<(f64,f64)>, //points that should be matched by the polynomial function
}

impl PolynomeEval
{
    pub fn new(points:Vec<(f64,f64)>) -> PolynomeEval
    {
        PolynomeEval { target: points }
    }
}

impl Evaluator for PolynomeEval
{
    //evaluate as inverted mean absolute error to target (we want to minimize instead of maximize)
    fn eval(&self, params:&Vec<f64>) -> f64
    {
        //calculate mean absolute error
        let mut mae = 0.0;
        for i in 0..self.target.len()
        {
            let point = &self.target[i];
            let error = point.1 - calc_f(params, point.0);
            mae += error.abs();
        }
        mae /= self.target.len() as f64;
        //regularize using L1
        let mut l1 = 0.0;
        for val in params.iter()
        {
            l1 += val.abs();
        }
        l1 /= params.len() as f64;
        l1 *= REGULARIZE;
        //return
        -(mae + l1)
    }
}
