//! Noisy polynomial example.
#![allow(clippy::missing_docs_in_private_items, clippy::print_stdout)]

use esopt::*;
use rand::prelude::*;

const DEGREE: usize = 3;
const REGULARIZE: Float = 0.025; //own L1 reg. factor

fn main() {
	//training data
	let points = vec![(0.0, 0.1), (1.0, 1.4), (2.0, 4.0), (3.0, 8.4), (4.0, 16.5)];
	/*let points = vec![    (0.0, 0.0),
		(1.0, 1.0),
		(2.0, 4.0),
		(3.0, 9.0),
		(4.0, 16.0)
	];*/

	//initialize required objects
	let mut opt = SGD::default(); //SGA optimizer (named SGD)
	opt.set_lr(0.1) //learning rate
		.set_beta(0.5) //momentum factor
		.set_lambda(0.01); //weight decay coefficient. we use our additional, own regularizer in the
				   // loss/objective function

	let eval = PolynomeEval::new(points); //target

	//create ES-Optimizer
	let mut es = ES::new(opt, eval); //Evolution-Strategy-Optimizer using optimizer and evaluator
	let params = gen_rnd_vec(DEGREE + 1, 0.1); //generate random initial parameters
	es.set_params(params); //initial parameters (important to specify the problem dimension, default is
					   // vec![0.0])

	//track the optimizer's results
	for i in 0..5 {
		let n = 200;
		let res = es.optimize_std_par(n); //optimize for n steps
		println!("After {:5} iterations:", (i + 1) * n);
		println!("-(MAE + L1): {:7.4}", res.0);
		println!("Gradnorm: {:7.5}", res.1);
		print_f(es.get_params());
		println!();
	}
}

fn calc_f(factors: &[Float], x: Float) -> Float {
	let mut result = factors[0];
	let mut current_pow = 1.0;
	for factor in factors.iter().skip(1) {
		current_pow *= x;
		result += factor * current_pow;
	}
	result
}

fn print_f(fct: &[Float]) {
	let mut str = String::new();
	for (i, factor) in fct.iter().enumerate() {
		str.push_str(&format!("{:7.4}x^{} + ", factor, i));
	}
	let tmp = str.len() - 3;
	str.truncate(tmp);
	println!("{}", str);
}

#[derive(Clone)]
struct PolynomeEval {
	target: Vec<(Float, Float)>, //points that should be matched by the polynomial function
}

impl PolynomeEval {
	pub fn new(points: Vec<(Float, Float)>) -> PolynomeEval {
		PolynomeEval { target: points }
	}
}

impl Evaluator for PolynomeEval {
	//evaluate as inverted mean absolute error to target (we want to minimize
	// instead of maximize)
	fn eval_train(&self, params: &[Float], _: usize) -> Float {
		let mut rng = thread_rng();
		//calculate mean absolute error of random batch of points -> seems to accept
		// outliers
		let mut mae = 0.0;
		let n = 4; //batch size
		for _ in 0..n {
			let i = rng.gen::<usize>() % self.target.len();
			let point = &self.target[i];
			let error = point.1 - calc_f(params, point.0);
			mae += error.abs();
		}
		mae /= n as Float;
		//regularize using L1
		let mut l1 = 0.0;
		for val in params.iter() {
			l1 += val.abs();
		}
		l1 /= params.len() as Float;
		l1 *= REGULARIZE;
		//return
		-(mae + l1)
	}

	//evaluate as inverted mean absolute error to target (we want to minimize
	// instead of maximize) here all data is used for deterministic progess
	// monitoring
	fn eval_test(&self, params: &[Float]) -> Float {
		//calculate mean absolute error
		let mut mae = 0.0;
		for i in 0..self.target.len() {
			let point = &self.target[i];
			let error = point.1 - calc_f(params, point.0);
			mae += error.abs();
		}
		mae /= self.target.len() as Float;
		//regularize using L1
		let mut l1 = 0.0;
		for val in params.iter() {
			l1 += val.abs();
		}
		l1 /= params.len() as Float;
		l1 *= REGULARIZE;
		//return
		-(mae + l1)
	}
}
