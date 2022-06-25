//! General implementation of the ES strategy described in https://arxiv.org/pdf/1703.03864.pdf.
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::expect_used)] // TODO: get rid of.

mod optimizers;

use std::cmp::Ordering;

pub use optimizers::*;
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;

/// This crate's float type to use.
pub type Float = f32;
#[cfg(feature = "floats-f64")]
/// This crate's float type to use.
pub type Float = f64;

// TODO:
// AdamaxBound ?
// DEBUG: show delta and grad?

/// Definition of standard evaluator trait.
pub trait Evaluator {
	/// Function to evaluate a set of parameters given as parameter.
	/// Return the score towards the target (optimizer maximizes).
	/// Only used once per optimization call (only for the returned score).
	fn eval_test(&self, parameters: &[Float]) -> Float;
	/// Function to evaluate a set of parameters also given the loop index as
	/// parameter. In addition to the parameters, the loop index is provided to
	/// allow selection of the same batch. Return the score towards the target
	/// (optimizer maximizes). Only used during training (very often).
	fn eval_train(&self, parameters: &[Float], loop_index: usize) -> Float;
}

/// Evolution-Strategy optimizer class. Optimizes given parameters towards a
/// maximum evaluation-score.
#[derive(Debug)]
pub struct ES<Feval: Evaluator, Opt: Optimizer> {
	/// Problem dimensionality.
	dim: usize,
	/// Current parameters.
	params: Vec<Float>,
	/// Chosen optimizer.
	opt: Opt,
	/// Evaluator function.
	eval: Feval,
	/// Standard deviation to calculate the noise for parameters.
	std: Float,
	/// Number of mirror-samples per step to approximate the gradient.
	samples: usize,
}

impl<Feval: Evaluator> ES<Feval, SGD> {
	/// Shortcut for ES::new(...) using SGD:
	/// Create a new ES-Optimizer using SGA (create SGD object with the given
	/// parameters).
	pub fn new_with_sgd(
		evaluator: Feval,
		learning_rate: Float,
		beta: Float,
		lambda: Float,
	) -> ES<Feval, SGD> {
		let mut optimizer = SGD::default();
		optimizer.set_lr(learning_rate).set_beta(beta).set_lambda(lambda);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}
}

impl<Feval: Evaluator> ES<Feval, Lookahead<SGD>> {
	/// Shortcut for ES::new(...) using Lookahead with SGD:
	/// Create a new ES-Optimizer using Lookahead with SGA (create Lookahead and
	/// SGD object with the given parameters).
	pub fn new_with_lookahead_sgd(
		evaluator: Feval,
		k: usize,
		learning_rate: Float,
		beta: Float,
		lambda: Float,
	) -> ES<Feval, Lookahead<SGD>> {
		let mut optimizer = SGD::default();
		optimizer.set_lr(learning_rate).set_beta(beta).set_lambda(lambda);
		let mut optimizer = Lookahead::new(optimizer);
		optimizer.set_k(k);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}
}

impl<Feval: Evaluator> ES<Feval, Adam> {
	/// Shortcut for ES::new(...) using Adam (/AdaBound):
	/// Create a new ES-Optimizer using Adam (create Adam object with the given
	/// parameters, rest left to default). Change these paramters using method
	/// get_opt_mut().set_<...>(...).
	pub fn new_with_adam(evaluator: Feval, learning_rate: Float, lambda: Float) -> ES<Feval, Adam> {
		let mut optimizer = Adam::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}

	/// Shortcut for ES::new(...) using Adam (/AdaBound):
	/// Create a new ES-Optimizer using Adam (create Adam object with the given
	/// parameters).
	pub fn new_with_adam_ex(
		evaluator: Feval,
		learning_rate: Float,
		lambda: Float,
		beta1: Float,
		beta2: Float,
		adabound: bool,
		final_lr: Float,
	) -> ES<Feval, Adam> {
		let mut optimizer = Adam::default();
		optimizer
			.set_lr(learning_rate)
			.set_lambda(lambda)
			.set_beta1(beta1)
			.set_beta2(beta2)
			.set_adabound(adabound)
			.set_final_lr(final_lr);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}
}

impl<Feval: Evaluator> ES<Feval, Lookahead<Adam>> {
	/// Shortcut for ES::new(...) using Lookahead with Adam:
	/// Create a new ES-Optimizer using Lookahead with Adam (create Lookahead
	/// and Adam object with the given parameters, rest left to default). Change
	/// these paramters using method get_opt_mut().set_<...>(...) and
	/// get_opt_mut().get_opt_mut().set_<...>(...).
	pub fn new_with_lookahead_adam(
		evaluator: Feval,
		k: usize,
		learning_rate: Float,
		lambda: Float,
	) -> ES<Feval, Lookahead<Adam>> {
		let mut optimizer = Adam::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda);
		let mut optimizer = Lookahead::new(optimizer);
		optimizer.set_k(k);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}

	/// Shortcut for ES::new(...) using Adam:
	/// Create a new ES-Optimizer using Adam (create Lookahead and Adam object
	/// with the given parameters).
	pub fn new_with_lookahead_adam_ex(
		evaluator: Feval,
		alpha: Float,
		k: usize,
		learning_rate: Float,
		lambda: Float,
		beta1: Float,
		beta2: Float,
	) -> ES<Feval, Lookahead<Adam>> {
		let mut optimizer = Adam::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda).set_beta1(beta1).set_beta2(beta2);
		let mut optimizer = Lookahead::new(optimizer);
		optimizer.set_alpha(alpha).set_k(k);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}
}

impl<Feval: Evaluator> ES<Feval, RAdam> {
	/// Shortcut for ES::new(...) using RAdam:
	/// Create a new ES-Optimizer using RAdam (create RAdam object with the
	/// given parameters, rest left to default). Change these paramters using
	/// method get_opt_mut().set_<...>(...).
	pub fn new_with_radam(
		evaluator: Feval,
		learning_rate: Float,
		lambda: Float,
	) -> ES<Feval, RAdam> {
		let mut optimizer = RAdam::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}

	/// Shortcut for ES::new(...) using RAdam:
	/// Create a new ES-Optimizer using RAdam (create RAdam object with the
	/// given parameters).
	pub fn new_with_radam_ex(
		evaluator: Feval,
		learning_rate: Float,
		lambda: Float,
		beta1: Float,
		beta2: Float,
	) -> ES<Feval, RAdam> {
		let mut optimizer = RAdam::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda).set_beta1(beta1).set_beta2(beta2);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}
}

impl<Feval: Evaluator> ES<Feval, Lookahead<RAdam>> {
	/// Shortcut for ES::new(...) using Lookahead with RAdam:
	/// Create a new ES-Optimizer using Lookahead with RAdam (create Lookahead
	/// and RAdam object with the given parameters, rest left to default).
	/// Change these paramters using method get_opt_mut().set_<...>(...) and
	/// get_opt_mut().get_opt_mut().set_<...>(...).
	pub fn new_with_lookahead_radam(
		evaluator: Feval,
		k: usize,
		learning_rate: Float,
		lambda: Float,
	) -> ES<Feval, Lookahead<RAdam>> {
		let mut optimizer = RAdam::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda);
		let mut optimizer = Lookahead::new(optimizer);
		optimizer.set_k(k);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}

	/// Shortcut for ES::new(...) using RAdam:
	/// Create a new ES-Optimizer using RAdam (create Lookahead and RAdam object
	/// with the given parameters).
	pub fn new_with_lookahead_radam_ex(
		evaluator: Feval,
		alpha: Float,
		k: usize,
		learning_rate: Float,
		lambda: Float,
		beta1: Float,
		beta2: Float,
	) -> ES<Feval, Lookahead<RAdam>> {
		let mut optimizer = RAdam::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda).set_beta1(beta1).set_beta2(beta2);
		let mut optimizer = Lookahead::new(optimizer);
		optimizer.set_alpha(alpha).set_k(k);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}
}

impl<Feval: Evaluator> ES<Feval, Adamax> {
	/// Shortcut for ES::new(...) using Adamax:
	/// Create a new ES-Optimizer using Adamax (create Adam object with the
	/// given parameters, rest left to default). Change these paramters using
	/// method get_opt_mut().set_<...>(...).
	pub fn new_with_adamax(
		evaluator: Feval,
		learning_rate: Float,
		lambda: Float,
	) -> ES<Feval, Adamax> {
		let mut optimizer = Adamax::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}

	/// Shortcut for ES::new(...) using Adam:
	/// Create a new ES-Optimizer using Adam (create Adam object with the given
	/// parameters).
	pub fn new_with_adamax_ex(
		evaluator: Feval,
		learning_rate: Float,
		lambda: Float,
		beta1: Float,
		beta2: Float,
		eps: Float,
	) -> ES<Feval, Adamax> {
		let mut optimizer = Adamax::default();
		optimizer
			.set_lr(learning_rate)
			.set_lambda(lambda)
			.set_beta1(beta1)
			.set_beta2(beta2)
			.set_eps(eps);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}
}

impl<Feval: Evaluator> ES<Feval, Lookahead<Adamax>> {
	/// Shortcut for ES::new(...) using Lookahead with Adamax:
	/// Create a new ES-Optimizer using Lookahead with Adamax (create Lookahead
	/// and Adamax object with the given parameters, rest left to default).
	/// Change these paramters using method get_opt_mut().set_<...>(...) and
	/// get_opt_mut().get_opt_mut().set_<...>(...).
	pub fn new_with_lookahead_adamax(
		evaluator: Feval,
		k: usize,
		learning_rate: Float,
		lambda: Float,
	) -> ES<Feval, Lookahead<Adamax>> {
		let mut optimizer = Adamax::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda);
		let mut optimizer = Lookahead::new(optimizer);
		optimizer.set_k(k);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}

	/// Shortcut for ES::new(...) using Adamax:
	/// Create a new ES-Optimizer using Adamax (create Lookahead and Adamax
	/// object with the given parameters).
	pub fn new_with_lookahead_adamax_ex(
		evaluator: Feval,
		alpha: Float,
		k: usize,
		learning_rate: Float,
		lambda: Float,
		beta1: Float,
		beta2: Float,
	) -> ES<Feval, Lookahead<Adamax>> {
		let mut optimizer = Adamax::default();
		optimizer.set_lr(learning_rate).set_lambda(lambda).set_beta1(beta1).set_beta2(beta2);
		let mut optimizer = Lookahead::new(optimizer);
		optimizer.set_alpha(alpha).set_k(k);
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}
}

impl<Feval: Evaluator, Opt: Optimizer> ES<Feval, Opt> {
	/// Create a new ES-Optimizer
	/// evaluator = object with Evaluator trait that computes the objetive-score
	/// based on the paramters optimizer = optimizer to calculate the parameter
	/// update using the gradient and the current parameters. (e.g. use
	/// SGD::new() aka SGA) Important: set the initial parameters afterswards by
	/// calling set_params to specify the problem dimension. (Default is [0.0],
	/// dim=1)
	pub fn new(optimizer: Opt, evaluator: Feval) -> ES<Feval, Opt> {
		ES { dim: 1, params: vec![0.0], opt: optimizer, eval: evaluator, std: 0.02, samples: 500 }
	}

	/// Set the parameters (potentially reinitializing the process)
	/// params = set of parameters to optimize
	pub fn set_params(&mut self, params: Vec<Float>) -> &mut Self {
		self.params = params;
		self.dim = self.params.len();

		self
	}

	/// Change the optimizer
	pub fn set_opt(&mut self, optimizer: Opt) -> &mut Self {
		self.opt = optimizer;

		self
	}

	/// Change the evaluator function
	pub fn set_eval(&mut self, evaluator: Feval) -> &mut Self {
		self.eval = evaluator;

		self
	}

	/// Set noise's standard deviation (applied to the parameters)
	/// Humanoid example in the paper used 0.02 as an example (default).
	/// Probably best to choose in dependence of evaluator output size.
	/// Tweak learning rate to fit to the std.
	pub fn set_std(&mut self, noise: Float) -> &mut Self {
		if noise <= 0.0 {
			panic!("Noise std may not be <= 0!");
		}
		self.std = noise;

		self
	}

	/// Set the number of mirror-samples per step to approximate the gradient
	/// Was probably around 700 in paper (1400 workers)
	pub fn set_samples(&mut self, num: usize) -> &mut Self {
		if num == 0 {
			panic!("Number of samples cannot be zero!");
		}
		self.samples = num;

		self
	}

	/// Get the current parameters (as ref)
	pub fn get_params(&self) -> &Vec<Float> {
		&self.params
	}

	/// Get the optimizer (as ref)
	pub fn get_opt(&self) -> &Opt {
		&self.opt
	}

	/// Get the evaluator (as ref)
	pub fn get_eval(&self) -> &Feval {
		&self.eval
	}

	/// Get the current parameters (as mut)
	pub fn get_params_mut(&mut self) -> &mut Vec<Float> {
		&mut self.params
	}

	/// Get the optimizer (as mut, to change parameters)
	pub fn get_opt_mut(&mut self) -> &mut Opt {
		&mut self.opt
	}

	/// Get the evaluator (as mut, to change parameters)
	pub fn get_eval_mut(&mut self) -> &mut Feval {
		&mut self.eval
	}

	/// Optimize for n steps.
	/// Uses the evaluator's score to calculate the gradients.
	/// Returns a tuple (score, gradnorm), which is the latest parameters'
	/// evaluated score and the norm of the last gradient/delta change.
	pub fn optimize(&mut self, n: usize) -> (Float, Float) {
		let mut rng = thread_rng();
		let mut grad = vec![0.0; self.dim];
		//for n iterations:
		let t = self.opt.get_t();
		for iterations in 0..n {
			//generate seed for repeatable random vector generation
			let seed = rng.gen::<u64>() % (std::u64::MAX - self.samples as u64);
			//approximate gradient with self.samples double-sided samples
			grad = vec![0.0; self.dim];
			for i in 0..self.samples {
				//(repeatable) eps generation
				let mut rng = SmallRng::seed_from_u64(seed + i as u64);
				//generate random epsilon
				let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
				//compute test parameters in both directions
				let mut testparampos = eps.clone();
				let mut testparamneg = eps.clone();
				for ((pos, neg), p) in
					testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
				{
					*pos = *p + *pos;
					*neg = *p - *neg;
				}
				//evaluate test parameters
				let scorepos = self.eval.eval_train(&testparampos, t + iterations);
				let scoreneg = self.eval.eval_train(&testparamneg, t + iterations);
				//calculate grad sum update
				for (g, e) in grad.iter_mut().zip(eps.iter()) {
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
	/// Returns a tuple (score, gradnorm), which is the latest parameters'
	/// evaluated score and the norm of the last gradient/delta change.
	pub fn optimize_ranked(&mut self, n: usize) -> (Float, Float) {
		let mut rng = thread_rng();
		let mut grad = vec![0.0; self.dim];
		//for n iterations:
		let t = self.opt.get_t();
		for iterations in 0..n {
			//generate seed for repeatable random vector generation
			let seed = rng.gen::<u64>() % (std::u64::MAX - self.samples as u64);
			//approximate gradient with self.samples double-sided samples
			grad = vec![0.0; self.dim];
			//first generate and fill whole vector of scores
			let mut scores = Vec::new();
			for i in 0..self.samples {
				//repeatable eps generation to save memory
				let mut rng = SmallRng::seed_from_u64(seed + i as u64);
				//gen and compute test parameters
				let mut testparampos = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
				let mut testparamneg = testparampos.clone();
				for ((pos, neg), p) in
					testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
				{
					*pos = *p + *pos;
					*neg = *p - *neg;
				}
				//evaluate parameters and save scores
				let scorepos = self.eval.eval_train(&testparampos, t + iterations);
				let scoreneg = self.eval.eval_train(&testparamneg, t + iterations);
				scores.push((i, false, scorepos));
				scores.push((i, true, scoreneg));
			}
			//sort, create ranks, sum up and calculate gradient from the sum
			sort_scores(&mut scores);
			scores.iter().enumerate().for_each(|(rank, (i, neg, _score))| {
				let mut rng = SmallRng::seed_from_u64(seed + *i as u64);
				let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
				let negfactor = if *neg { -1.0 } else { 1.0 };
				let centered_rank = rank as Float / (self.samples as Float - 0.5) - 1.0;
				for (g, e) in grad.iter_mut().zip(eps.iter()) {
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
	/// Returns a tuple (score, gradnorm), which is the latest parameters'
	/// evaluated score and the norm of the last gradient/delta change.
	pub fn optimize_std(&mut self, n: usize) -> (Float, Float) {
		let mut rng = thread_rng();
		let mut grad = vec![0.0; self.dim];
		//for n iterations:
		let t = self.opt.get_t();
		for iterations in 0..n {
			//generate seed for repeatable random vector generation
			let seed = rng.gen::<u64>() % (std::u64::MAX - self.samples as u64);
			//approximate gradient with self.samples double-sided samples
			grad = vec![0.0; self.dim];
			//first generate and fill whole vector of scores
			let mut scores = vec![(0.0, 0.0); self.samples];
			scores.iter_mut().enumerate().for_each(|(i, (scorepos, scoreneg))| {
				//repeatable eps generation to save memory
				let mut rng = SmallRng::seed_from_u64(seed + i as u64);
				//gen and compute test parameters
				let mut testparampos = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
				let mut testparamneg = testparampos.clone();
				for ((pos, neg), p) in
					testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
				{
					*pos = *p + *pos;
					*neg = *p - *neg;
				}
				//evaluate parameters and save scores
				*scorepos = self.eval.eval_train(&testparampos, t + iterations);
				*scoreneg = self.eval.eval_train(&testparamneg, t + iterations);
			});
			//calculate std, mean
			let (_mean, std) = get_mean_std(&scores);
			//sum up and calculate gradient from the sum
			scores.iter().enumerate().for_each(|(i, (scorepos, scoreneg))| {
				let mut rng = SmallRng::seed_from_u64(seed + i as u64);
				let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
				for (g, e) in grad.iter_mut().zip(eps.iter()) {
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
	/// Returns a tuple (score, gradnorm), which is the latest parameters'
	/// evaluated score and the norm of the last gradient/delta change.
	pub fn optimize_norm(&mut self, n: usize) -> (Float, Float) {
		let mut rng = thread_rng();
		let mut grad = vec![0.0; self.dim];
		//for n iterations:
		let t = self.opt.get_t();
		for iterations in 0..n {
			//generate seed for repeatable random vector generation
			let seed = rng.gen::<u64>() % (std::u64::MAX - self.samples as u64);
			//approximate gradient with self.samples double-sided samples
			grad = vec![0.0; self.dim];
			//first generate and fill whole vector of scores
			let mut scores = vec![(0.0, 0.0); self.samples];
			let mut maximum = -1.0;
			scores.iter_mut().enumerate().for_each(|(i, (scorepos, scoreneg))| {
				//repeatable eps generation to save memory
				let mut rng = SmallRng::seed_from_u64(seed + i as u64);
				//gen and compute test parameters
				let mut testparampos = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
				let mut testparamneg = testparampos.clone();
				for ((pos, neg), p) in
					testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
				{
					*pos = *p + *pos;
					*neg = *p - *neg;
				}
				//evaluate parameters and save scores
				*scorepos = self.eval.eval_train(&testparampos, t + iterations);
				*scoreneg = self.eval.eval_train(&testparamneg, t + iterations);
				//calculate maxmimum absolute score
				if scorepos.abs() > maximum {
					maximum = scorepos.abs();
				}
				if scoreneg.abs() > maximum {
					maximum = scoreneg.abs();
				}
			});
			//sum up and calculate gradient from the sum
			scores.iter().enumerate().for_each(|(i, (scorepos, scoreneg))| {
				let mut rng = SmallRng::seed_from_u64(seed + i as u64);
				let eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
				for (g, e) in grad.iter_mut().zip(eps.iter()) {
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
	/// Returns a tuple (score, gradnorm), which is the latest parameters'
	/// evaluated score and the norm of the last gradient/delta change.
	pub fn optimize_par(&mut self, n: usize) -> (Float, Float)
	where
		Opt: Sync,
		Feval: Sync,
	{
		let mut rng = thread_rng();
		let mut grad = vec![0.0; self.dim];
		//for n iterations:
		let t = self.opt.get_t();
		for iterations in 0..n {
			//generate seed for repeatable random vector generation
			let seed = rng.gen::<u64>() % (std::u64::MAX - self.samples as u64);
			//approximate gradient with self.samples double-sided samples
			grad = (0..self.samples)
				.into_par_iter()
				.map(|i| {
					//(repeatable) eps generation
					let mut rng = SmallRng::seed_from_u64(seed + i as u64);
					//gen and compute test parameters
					let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
					let mut testparampos = eps.clone();
					let mut testparamneg = eps.clone();
					for ((pos, neg), p) in
						testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
					{
						*pos = *p + *pos;
						*neg = *p - *neg;
					}
					//evaluate parameters to compute scores
					let scorepos = self.eval.eval_train(&testparampos, t + iterations);
					let scoreneg = self.eval.eval_train(&testparamneg, t + iterations);
					//compute gradient parts and the sum up in reduce to calculate the gradient
					mul_scalar(&mut eps, scorepos - scoreneg);
					eps
				})
				.reduce(
					|| vec![0.0; self.dim],
					|mut a, b| {
						add_inplace(&mut a, &b);
						a
					},
				);
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
	/// Returns a tuple (score, gradnorm), which is the latest parameters'
	/// evaluated score and the norm of the last gradient/delta change.
	pub fn optimize_ranked_par(&mut self, n: usize) -> (Float, Float)
	where
		Opt: Sync,
		Feval: Sync,
	{
		let mut rng = thread_rng();
		let mut grad = vec![0.0; self.dim];
		//for n iterations:
		let t = self.opt.get_t();
		for iterations in 0..n {
			//generate seed for repeatable random vector generation
			let seed = rng.gen::<u64>() % (std::u64::MAX - self.samples as u64);
			//approximate gradient with self.samples double-sided samples
			//first generate and fill whole vector of scores
			let mut scores = vec![(0, false, 0.0); 2 * self.samples];
			for i in 0..self.samples {
				scores[2 * i].0 = i;
				scores[2 * i + 1].0 = i;
				scores[2 * i + 1].1 = true;
			}
			scores.par_iter_mut().for_each(|(i, neg, score)| {
				//repeatable eps generation to save memory
				let mut rng = SmallRng::seed_from_u64(seed + *i as u64);
				//gen and compute test parameters
				let mut testparam = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
				if *neg {
					mul_scalar(&mut testparam, -1.0);
				}
				add_inplace(&mut testparam, &self.params);
				//evaluate parameters and save scores
				*score = self.eval.eval_train(&testparam, t + iterations);
			});
			//compute the centered ranks and calculate the summed result to compute the
			// gradient
			sort_scores(&mut scores);
			grad = scores
				.par_iter()
				.enumerate()
				.map(|(rank, (i, neg, _score))| {
					let mut rng = SmallRng::seed_from_u64(seed + *i as u64);
					let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
					let negfactor = if *neg { -1.0 } else { 1.0 };
					let centered_rank = rank as Float / (self.samples as Float - 0.5) - 1.0;
					mul_scalar(&mut eps, negfactor * centered_rank);
					eps
				})
				.reduce(
					|| vec![0.0; self.dim],
					|mut a, b| {
						add_inplace(&mut a, &b);
						a
					},
				);
			//if reduce saves too much and takes too much memory: do serial (normal iter)
			// and initialize grad before, sum components to grad in loop (for_each);
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
	/// Returns a tuple (score, gradnorm), which is the latest parameters'
	/// evaluated score and the norm of the last gradient/delta change.
	pub fn optimize_std_par(&mut self, n: usize) -> (Float, Float)
	where
		Opt: Sync,
		Feval: Sync,
	{
		let mut rng = thread_rng();
		let mut grad = vec![0.0; self.dim];
		//for n iterations:
		let t = self.opt.get_t();
		for iterations in 0..n {
			//generate seed for repeatable random vector generation
			let seed = rng.gen::<u64>() % (std::u64::MAX - self.samples as u64);
			//approximate gradient with self.samples double-sided samples
			//first generate and fill whole vector of scores
			let mut scores = vec![(0.0, 0.0); self.samples];
			scores.par_iter_mut().enumerate().for_each(|(i, (scorepos, scoreneg))| {
				//repeatable eps generation to save memory
				let mut rng = SmallRng::seed_from_u64(seed + i as u64);
				//gen and compute test parameters
				let mut testparampos = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
				let mut testparamneg = testparampos.clone();
				for ((pos, neg), p) in
					testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
				{
					*pos = *p + *pos;
					*neg = *p - *neg;
				}
				//evaluate parameters and save scores
				*scorepos = self.eval.eval_train(&testparampos, t + iterations);
				*scoreneg = self.eval.eval_train(&testparamneg, t + iterations);
			});
			//calculate std, mean
			let (_mean, std) = get_mean_std(&scores);
			//sum up and calculate gradient from the sum
			grad = scores
				.par_iter()
				.enumerate()
				.map(|(i, (scorepos, scoreneg))| {
					let mut rng = SmallRng::seed_from_u64(seed + i as u64);
					let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
					//subtraction by mean cancels out
					mul_scalar(&mut eps, (*scorepos - *scoreneg) / std);
					eps
				})
				.reduce(
					|| vec![0.0; self.dim],
					|mut a, b| {
						add_inplace(&mut a, &b);
						a
					},
				);
			//if reduce saves too much and takes too much memory: do serial (normal iter)
			// and initialize grad before, sum components to grad in loop (for_each);
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
	/// Returns a tuple (score, gradnorm), which is the latest parameters'
	/// evaluated score and the norm of the last gradient/delta change.
	pub fn optimize_norm_par(&mut self, n: usize) -> (Float, Float)
	where
		Opt: Sync,
		Feval: Sync,
	{
		let mut rng = thread_rng();
		let mut grad = vec![0.0; self.dim];
		//for n iterations:
		let t = self.opt.get_t();
		for iterations in 0..n {
			//generate seed for repeatable random vector generation
			let seed = rng.gen::<u64>() % (std::u64::MAX - self.samples as u64);
			//approximate gradient with self.samples double-sided samples
			//first generate and fill whole vector of scores
			let mut scores = vec![(0.0, 0.0); self.samples];
			scores.par_iter_mut().enumerate().for_each(|(i, (scorepos, scoreneg))| {
				//repeatable eps generation to save memory
				let mut rng = SmallRng::seed_from_u64(seed + i as u64);
				//gen and compute test parameters
				let mut testparampos = gen_rnd_vec_rng(&mut rng, self.dim, self.std); //eps
				let mut testparamneg = testparampos.clone();
				for ((pos, neg), p) in
					testparampos.iter_mut().zip(testparamneg.iter_mut()).zip(self.params.iter())
				{
					*pos = *p + *pos;
					*neg = *p - *neg;
				}
				//evaluate parameters and save scores
				*scorepos = self.eval.eval_train(&testparampos, t + iterations);
				*scoreneg = self.eval.eval_train(&testparamneg, t + iterations);
			});
			//calculate maxmimum absolute score
			let mut maximum = -1.0;
			scores.iter().for_each(|x| {
				if x.0.abs() > maximum {
					maximum = x.0.abs();
				}
				if x.1.abs() > maximum {
					maximum = x.1.abs();
				}
			});
			//sum up and calculate gradient from the sum
			grad = scores
				.par_iter()
				.enumerate()
				.map(|(i, (scorepos, scoreneg))| {
					let mut rng = SmallRng::seed_from_u64(seed + i as u64);
					let mut eps = gen_rnd_vec_rng(&mut rng, self.dim, self.std);
					//subtraction by mean cancels out
					mul_scalar(&mut eps, (*scorepos - *scoreneg) / maximum);
					eps
				})
				.reduce(
					|| vec![0.0; self.dim],
					|mut a, b| {
						add_inplace(&mut a, &b);
						a
					},
				);
			//if reduce saves too much and takes too much memory: do serial (normal iter)
			// and initialize grad before, sum components to grad in loop (for_each);
			mul_scalar(&mut grad, 1.0 / ((2 * self.samples) as Float * self.std));
			//calculate the delta update using the optimizer
			let delta = self.opt.get_delta(&self.params, &grad);
			//update the parameters
			add_inplace(&mut self.params, &delta);
		}

		(self.eval.eval_test(&self.params), norm(&grad))
	}
}

/// Generate a vector of random numbers with 0 mean and std std, normally
/// distributed. Using specified RNG.
fn gen_rnd_vec_rng<RNG: Rng>(rng: &mut RNG, n: usize, std: Float) -> Vec<Float> {
	let normal =
		Normal::new(0.0, f64::from(std)).expect("Invalid parameters for Normal distribution!");
	normal.sample_iter(rng).take(n).map(|x| x as Float).collect()
}

/// Generate a vector of random numbers with 0 mean and std std, normally
/// distributed. Using standard thread_rng.
#[must_use]
pub fn gen_rnd_vec(n: usize, std: Float) -> Vec<Float> {
	let mut rng = thread_rng();
	let normal =
		Normal::new(0.0, f64::from(std)).expect("Invalid parameters for Normal distribution!");
	normal.sample_iter(&mut rng).take(n).map(|x| x as Float).collect()
}

/// Add a second vector onto the first vector in place
fn add_inplace(v1: &mut [Float], v2: &[Float]) {
	for (val1, val2) in v1.iter_mut().zip(v2.iter()) {
		*val1 += *val2;
	}
}

/// Multiplies a scalar to a vector
fn mul_scalar(vec: &mut [Float], scalar: Float) {
	for val in vec.iter_mut() {
		*val *= scalar;
	}
}

/// Calculates the norm of a vector
#[must_use]
fn norm(vec: &[Float]) -> Float {
	let mut norm = 0.0;
	for val in vec.iter() {
		norm += *val * *val;
	}
	norm.sqrt()
}

/// calculate mean and standard deviation of the scores
#[must_use]
fn get_mean_std(vec: &[(Float, Float)]) -> (Float, Float) {
	let mut mean = 0.0;
	vec.iter().for_each(|(scorepos, scoreneg)| {
		mean += *scorepos + *scoreneg;
	});
	mean /= (2 * vec.len()) as Float;

	let mut std = 0.0;
	vec.iter().for_each(|(scorepos, scoreneg)| {
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
fn sort_scores<T, U>(vec: &mut [(T, U, Float)]) {
	//worst score in front
	vec.sort_unstable_by(|r1, r2| {
		//partial cmp and check for NaN
		(r1.2).partial_cmp(&r2.2).unwrap_or_else(|| {
			if r1.2.is_nan() {
				if r2.2.is_nan() {
					Ordering::Equal
				} else {
					Ordering::Less
				}
			} else {
				Ordering::Greater
			}
		})
	});
}
