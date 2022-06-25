//! Implementation of general gradient optimizers.

use std::{fs, fs::File, io::prelude::*};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::Float;

/// Definition of the optimizer traits, to dynamically allow different
/// optimizers
pub trait Optimizer {
	/// Function to compute the delta step/update later applied to the
	/// parameters Takes parameters and gradient as input
	/// Returns delta vector
	fn get_delta(&mut self, parameters: &[Float], gradient: &[Float]) -> Vec<Float>;
	/// Get number of iterations already processed.
	fn get_t(&self) -> usize;
}

/// SGD Optimizer, which actually is SGA here (stochastic gradient ascent)
/// Momentum and weight decay is available
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGD {
	/// Learning rate.
	lr: Float,
	/// Weight decay coefficient.
	lambda: Float,
	/// Momentum coefficient.
	beta: Float,
	/// Last momentum gradient.
	lastv: Vec<Float>,
	/// Number of iterations/timesteps.
	t: usize,
}

impl Default for SGD {
	fn default() -> Self {
		Self { lr: 0.01, lambda: 0.0, beta: 0.0, lastv: vec![0.0], t: 0 }
	}
}

impl SGD {
	/// Set learning rate
	pub fn set_lr(&mut self, learning_rate: Float) -> &mut Self {
		if learning_rate <= 0.0 {
			panic!("Learning rate must be greater than zero!");
		}
		self.lr = learning_rate;

		self
	}

	/// Set lambda factor for weight decay
	pub fn set_lambda(&mut self, coeff: Float) -> &mut Self {
		if coeff < 0.0 {
			panic!("Lambda coefficient may not be smaller than zero!");
		}
		self.lambda = coeff;

		self
	}

	/// Set beta factor for momentum
	pub fn set_beta(&mut self, factor: Float) -> &mut Self {
		if !(0.0..1.0).contains(&factor) {
			panic!("Prohibited momentum paramter: {}. Must be in [0.0, 1.0)!", factor);
		}
		self.beta = factor;

		self
	}

	/// Encodes the optimizer as a JSON string.
	#[must_use]
	pub fn to_json(&self) -> String {
		serde_json::to_string(self).expect("Encoding JSON failed!")
	}

	/// Builds a new optimizer from a JSON string.
	#[must_use]
	pub fn from_json(encoded: &str) -> SGD {
		serde_json::from_str(encoded).expect("Decoding JSON failed!")
	}

	/// Saves the model to a file
	pub fn save(&self, file: &str) -> Result<(), std::io::Error> {
		let mut file = File::create(file)?;
		let json = self.to_json();
		file.write_all(json.as_bytes())?;
		Ok(())
	}

	/// Creates a model from a previously saved file
	pub fn load(file: &str) -> Result<SGD, std::io::Error> {
		let json = fs::read_to_string(file)?;
		Ok(SGD::from_json(&json))
	}
}

impl Optimizer for SGD {
	/// Compute delta update from params and gradient
	fn get_delta(&mut self, params: &[Float], grad: &[Float]) -> Vec<Float> {
		if self.lastv.len() != params.len() {
			//initialize with zero gradient
			self.lastv = vec![0.0; params.len()];
		}

		//calculate momentum update and compute delta (parameter update)
		let mut delta = grad.to_vec();
		for ((m, d), p) in self.lastv.iter_mut().zip(delta.iter_mut()).zip(params.iter()) {
			//momentum update
			*m = self.beta.mul_add(*m, (1.0 - self.beta) * *d);
			//compute delta based on momentum
			*d = self.lr * *m; //here no minus, because ascend instead of descent
				   //add weight decay
			*d -= self.lr * self.lambda * *p;
		}
		self.t += 1;

		//return
		delta
	}

	/// Retrieve the timestep (to allow computing manual learning rate decay)
	fn get_t(&self) -> usize {
		self.t
	}
}

/// Adam Optimizer, with possibility of using AdaBound
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
	/// learning rate.
	lr: Float,
	/// weight decay coefficient.
	lambda: Float,
	/// exponential moving average factor.
	beta1: Float,
	/// exponential second moment average factor (squared gradient).
	beta2: Float,
	/// small epsilon to avoid divide by zero (fuzz factor).
	eps: Float,
	/// number of taken timesteps.
	t: usize,
	/// first order moment (avg).
	avggrad1: Vec<Float>,
	/// second oder moment (squared).
	avggrad2: Vec<Float>,
	/// switch whether to use the AdaBound variant.
	adabound: bool,
	/// final LR to use using AdaBound (SGD).
	final_lr: Float,
	/// convergence speed of bounding functions for AdaBound.
	gamma: Float,
}

impl Default for Adam {
	/// Create new Adam optimizer instance using default hyperparameters (lr =
	/// 0.001, lambda = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, adabound =
	/// false, final_lr = 0.1, gamma: 0.001) Also try higher LR; beta2 = 0.99;
	/// try adabound!
	fn default() -> Self {
		Self {
			lr: 0.001,
			lambda: 0.0,
			beta1: 0.9,
			beta2: 0.999,
			eps: 1e-8,
			t: 0,
			avggrad1: vec![0.0],
			avggrad2: vec![0.0],
			adabound: false,
			final_lr: 0.1,
			gamma: 0.001,
		}
	}
}

impl Adam {
	/// Set learning rate
	pub fn set_lr(&mut self, learning_rate: Float) -> &mut Self {
		if learning_rate <= 0.0 {
			panic!("Learning rate must be greater than zero!");
		}
		self.lr = learning_rate;

		self
	}

	/// Set final learning rate for AdaBound (SGD)
	pub fn set_final_lr(&mut self, learning_rate: Float) -> &mut Self {
		if learning_rate <= 0.0 {
			panic!("Learning rate must be greater than zero!");
		}
		self.final_lr = learning_rate;

		self
	}

	/// Set lambda factor for weight decay
	pub fn set_lambda(&mut self, coeff: Float) -> &mut Self {
		if coeff < 0.0 {
			panic!("Lambda coefficient may not be smaller than zero!");
		}
		self.lambda = coeff;

		self
	}

	/// Set gamma factor for AdaBound bounding convergence
	pub fn set_gamma(&mut self, coeff: Float) -> &mut Self {
		if !(0.0..1.0).contains(&coeff) {
			panic!("Gamma coefficient is in appropriate!");
		}
		self.gamma = coeff;

		self
	}

	/// Set beta1 coefficient (for exponential moving average of first moment)
	pub fn set_beta1(&mut self, beta: Float) -> &mut Self {
		if !(0.0..1.0).contains(&beta) {
			panic!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta);
		}
		self.beta1 = beta;

		self
	}

	/// Set beta2 coefficient (for exponential moving average of second moment)
	pub fn set_beta2(&mut self, beta: Float) -> &mut Self {
		if !(0.0..1.0).contains(&beta) {
			panic!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta);
		}
		self.beta2 = beta;

		self
	}

	/// Set epsilon to avoid divide by zero (fuzz factor)
	pub fn set_eps(&mut self, epsilon: Float) -> &mut Self {
		if epsilon < 0.0 {
			panic!("Epsilon must be >= 0!");
		}
		self.eps = epsilon;

		self
	}

	/// Set usage of AdaBound
	pub fn set_adabound(&mut self, use_bound: bool) -> &mut Self {
		self.adabound = use_bound;
		self
	}

	/// Encodes the optimizer as a JSON string.
	#[must_use]
	pub fn to_json(&self) -> String {
		serde_json::to_string(self).expect("Encoding JSON failed!")
	}

	/// Builds a new optimizer from a JSON string.
	#[must_use]
	pub fn from_json(encoded: &str) -> Adam {
		serde_json::from_str(encoded).expect("Decoding JSON failed!")
	}

	/// Saves the model to a file
	pub fn save(&self, file: &str) -> Result<(), std::io::Error> {
		let mut file = File::create(file)?;
		let json = self.to_json();
		file.write_all(json.as_bytes())?;
		Ok(())
	}

	/// Creates a model from a previously saved file
	pub fn load(file: &str) -> Result<Adam, std::io::Error> {
		let json = fs::read_to_string(file)?;
		Ok(Adam::from_json(&json))
	}
}

impl Optimizer for Adam {
	/// Compute delta update from params and gradient
	fn get_delta(&mut self, params: &[Float], grad: &[Float]) -> Vec<Float> {
		if self.avggrad1.len() != params.len() || self.avggrad2.len() != params.len() {
			//initialize with zero moments
			self.avggrad1 = vec![0.0; params.len()];
			self.avggrad2 = vec![0.0; params.len()];
		}

		//timestep + unbias factor
		self.t += 1;
		let lr_unbias = self.lr * (1.0 - self.beta2.powf(self.t as Float)).sqrt()
			/ (1.0 - self.beta1.powf(self.t as Float));
		//dynamic bound
		let lower_bound = (1.0 - 1.0 / (self.gamma.mul_add(self.t as Float, 1.0))) * self.final_lr;
		let upper_bound = (1.0 + 1.0 / (self.gamma * self.t as Float)) * self.final_lr;

		//update exponential moving averages and compute delta (parameter update)
		let mut delta = grad.to_vec();
		for (((g1, g2), d), p) in self
			.avggrad1
			.iter_mut()
			.zip(self.avggrad2.iter_mut())
			.zip(delta.iter_mut())
			.zip(params.iter())
		{
			//moment 1 and 2 update
			*g1 = self.beta1.mul_add(*g1, (1.0 - self.beta1) * *d);
			*g2 = self.beta2.mul_add(*g2, (1.0 - self.beta2) * *d * *d);
			//delta update
			if self.adabound {
				//dynamic bound
				let bound_lr =
					(lr_unbias / (g2.sqrt() + self.eps)).max(lower_bound).min(upper_bound);
				*d = bound_lr * *g1;
			} else {
				*d = lr_unbias * *g1 / (g2.sqrt() + self.eps); //normally it would be
				                               // -lr_unbias, but we want to
				                               // maximize
			}
			//weight decay
			*d -= self.lr * self.lambda * *p;
		}

		//return
		delta
	}

	/// Retrieve the timestep (to allow computing manual learning rate decay)
	fn get_t(&self) -> usize {
		self.t
	}
}

/// RAdam Optimizer (Rectified Adam)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAdam {
	/// learning rate.
	lr: Float,
	/// weight decay coefficient.
	lambda: Float,
	/// exponential moving average factor.
	beta1: Float,
	/// exponential second moment average factor (squared gradient).
	beta2: Float,
	/// small epsilon to avoid divide by zero (fuzz factor).
	eps: Float,
	/// number of taken timesteps.
	t: usize,
	/// first order moment (avg).
	avggrad1: Vec<Float>,
	/// second oder moment (squared).
	avggrad2: Vec<Float>,
}

impl Default for RAdam {
	/// Create new RAdam optimizer instance using default hyperparameters (lr =
	/// 0.001, lambda = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
	/// Also try higher LR; beta2 = 0.99; try adabound!
	fn default() -> Self {
		RAdam {
			lr: 0.001,
			lambda: 0.0,
			beta1: 0.9,
			beta2: 0.999,
			eps: 1e-8,
			t: 0,
			avggrad1: vec![0.0],
			avggrad2: vec![0.0],
		}
	}
}

impl RAdam {
	/// Set learning rate
	pub fn set_lr(&mut self, learning_rate: Float) -> &mut Self {
		if learning_rate <= 0.0 {
			panic!("Learning rate must be greater than zero!");
		}
		self.lr = learning_rate;

		self
	}

	/// Set lambda factor for weight decay
	pub fn set_lambda(&mut self, coeff: Float) -> &mut Self {
		if coeff < 0.0 {
			panic!("Lambda coefficient may not be smaller than zero!");
		}
		self.lambda = coeff;

		self
	}

	/// Set beta1 coefficient (for exponential moving average of first moment)
	pub fn set_beta1(&mut self, beta: Float) -> &mut Self {
		if !(0.0..1.0).contains(&beta) {
			panic!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta);
		}
		self.beta1 = beta;

		self
	}

	/// Set beta2 coefficient (for exponential moving average of second moment)
	pub fn set_beta2(&mut self, beta: Float) -> &mut Self {
		if !(0.0..1.0).contains(&beta) {
			panic!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta);
		}
		self.beta2 = beta;

		self
	}

	/// Set epsilon to avoid divide by zero (fuzz factor)
	pub fn set_eps(&mut self, epsilon: Float) -> &mut Self {
		if epsilon < 0.0 {
			panic!("Epsilon must be >= 0!");
		}
		self.eps = epsilon;

		self
	}

	/// Encodes the optimizer as a JSON string.
	#[must_use]
	pub fn to_json(&self) -> String {
		serde_json::to_string(self).expect("Encoding JSON failed!")
	}

	/// Builds a new optimizer from a JSON string.
	#[must_use]
	pub fn from_json(encoded: &str) -> RAdam {
		serde_json::from_str(encoded).expect("Decoding JSON failed!")
	}

	/// Saves the model to a file
	pub fn save(&self, file: &str) -> Result<(), std::io::Error> {
		let mut file = File::create(file)?;
		let json = self.to_json();
		file.write_all(json.as_bytes())?;
		Ok(())
	}

	/// Creates a model from a previously saved file
	pub fn load(file: &str) -> Result<RAdam, std::io::Error> {
		let json = fs::read_to_string(file)?;
		Ok(RAdam::from_json(&json))
	}
}

impl Optimizer for RAdam {
	/// Compute delta update from params and gradient
	fn get_delta(&mut self, params: &[Float], grad: &[Float]) -> Vec<Float> {
		if self.avggrad1.len() != params.len() || self.avggrad2.len() != params.len() {
			//initialize with zero moments
			self.avggrad1 = vec![0.0; params.len()];
			self.avggrad2 = vec![0.0; params.len()];
		}

		//timestep, bias-correct LR, SMAs for rectification
		self.t += 1;
		let t_float = self.t as Float;
		let beta1_pt = self.beta1.powf(t_float);
		let beta2_pt = self.beta2.powf(t_float);
		let sma_inf = 2.0 / (1.0 - self.beta2) - 1.0;
		let sma_t = sma_inf - 2.0 * t_float * beta2_pt / (1.0 - beta2_pt);
		let r_t = (((sma_t - 4.0) * (sma_t - 2.0) * sma_inf)
			/ ((sma_inf - 4.0) * (sma_inf - 2.0) * sma_t))
			.sqrt(); //variance rectification term
		let lr_unbias1 = self.lr / (1.0 - beta1_pt);
		let lr_unbias12 = self.lr * (1.0 - beta2_pt).sqrt() / (1.0 - beta1_pt);

		//update exponential moving averages and compute delta (parameter update)
		let mut delta = grad.to_vec();
		for (((g1, g2), d), p) in self
			.avggrad1
			.iter_mut()
			.zip(self.avggrad2.iter_mut())
			.zip(delta.iter_mut())
			.zip(params.iter())
		{
			//moment 1 and 2 update
			*g1 = self.beta1.mul_add(*g1, (1.0 - self.beta1) * *d);
			*g2 = self.beta2.mul_add(*g2, (1.0 - self.beta2) * *d * *d);
			//delta update depending on variance
			if sma_t > 4.0 {
				*d = lr_unbias12 * r_t * *g1 / (g2.sqrt() + self.eps); //normally it would be
				                                       // -lr_unbias, but we
				                                       // want to maximize
			} else {
				*d = lr_unbias1 * *g1; //normally it would be -lr_unbias, but we want to
				       // maximize
			}
			//weight decay
			*d -= self.lr * self.lambda * *p;
		}

		//return
		delta
	}

	/// Retrieve the timestep (to allow computing manual learning rate decay)
	fn get_t(&self) -> usize {
		self.t
	}
}

/// Adamax Optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adamax {
	/// learning rate.
	lr: Float,
	/// weight decay coefficient.
	lambda: Float,
	/// exponential moving average factor.
	beta1: Float,
	/// exponential second moment average factor (squared gradient).
	beta2: Float,
	/// small epsilon to avoid divide by zero (fuzz factor).
	eps: Float,
	/// number of taken timesteps.
	t: usize,
	/// first order moment (avg).
	avggrad1: Vec<Float>,
	/// second oder moment (squared).
	avggrad2: Vec<Float>,
}

impl Default for Adamax {
	/// Create new Adamax optimizer instance using default hyperparameters (lr =
	/// 0.002, lambda = 0, beta1 = 0.9, beta2 = 0.999, eps = 0) Also try higher
	/// LR; beta2 = 0.99
	fn default() -> Self {
		Self {
			lr: 0.002,
			lambda: 0.0,
			beta1: 0.9,
			beta2: 0.999,
			eps: 0.0,
			t: 0,
			avggrad1: vec![0.0],
			avggrad2: vec![0.0],
		}
	}
}

impl Adamax {
	/// Set learning rate
	pub fn set_lr(&mut self, learning_rate: Float) -> &mut Self {
		if learning_rate <= 0.0 {
			panic!("Learning rate must be greater than zero!");
		}
		self.lr = learning_rate;

		self
	}

	/// Set lambda factor for weight decay
	pub fn set_lambda(&mut self, coeff: Float) -> &mut Self {
		if coeff < 0.0 {
			panic!("Lambda coefficient may not be smaller than zero!");
		}
		self.lambda = coeff;

		self
	}

	/// Set beta1 coefficient (for exponential moving average of first moment)
	pub fn set_beta1(&mut self, beta: Float) -> &mut Self {
		if !(0.0..1.0).contains(&beta) {
			panic!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta);
		}
		self.beta1 = beta;

		self
	}

	/// Set beta2 coefficient (for exponential moving average of second moment)
	pub fn set_beta2(&mut self, beta: Float) -> &mut Self {
		if !(0.0..1.0).contains(&beta) {
			panic!("Prohibited beta coefficient: {}. Must be in [0.0, 1.0)!", beta);
		}
		self.beta2 = beta;

		self
	}

	/// Set epsilon to avoid divide by zero (fuzz factor)
	pub fn set_eps(&mut self, epsilon: Float) -> &mut Self {
		if epsilon < 0.0 {
			panic!("Epsilon must be >= 0!");
		}
		self.eps = epsilon;

		self
	}

	/// Encodes the optimizer as a JSON string.
	#[must_use]
	pub fn to_json(&self) -> String {
		serde_json::to_string(self).expect("Encoding JSON failed!")
	}

	/// Builds a new optimizer from a JSON string.
	#[must_use]
	pub fn from_json(encoded: &str) -> Adamax {
		serde_json::from_str(encoded).expect("Decoding JSON failed!")
	}

	/// Saves the model to a file
	pub fn save(&self, file: &str) -> Result<(), std::io::Error> {
		let mut file = File::create(file)?;
		let json = self.to_json();
		file.write_all(json.as_bytes())?;
		Ok(())
	}

	/// Creates a model from a previously saved file
	pub fn load(file: &str) -> Result<Adamax, std::io::Error> {
		let json = fs::read_to_string(file)?;
		Ok(Adamax::from_json(&json))
	}
}

impl Optimizer for Adamax {
	/// Compute delta update from params and gradient
	fn get_delta(&mut self, params: &[Float], grad: &[Float]) -> Vec<Float> {
		if self.avggrad1.len() != params.len() || self.avggrad2.len() != params.len() {
			//initialize with zero moments
			self.avggrad1 = vec![0.0; params.len()];
			self.avggrad2 = vec![0.0; params.len()];
		}

		//timestep + unbias factor
		self.t += 1;
		let lr_unbias = self.lr / (1.0 - self.beta1.powf(self.t as Float));

		//update exponential moving averages and compute delta (parameter update)
		let mut delta = grad.to_vec();
		for (((g1, g2), d), p) in self
			.avggrad1
			.iter_mut()
			.zip(self.avggrad2.iter_mut())
			.zip(delta.iter_mut())
			.zip(params.iter())
		{
			//moment 1 and 2 update
			*g1 = self.beta1.mul_add(*g1, (1.0 - self.beta1) * *d);
			*g2 = (self.beta2 * *g2).max(d.abs());
			//delta update
			*d = lr_unbias * *g1 / (*g2 + self.eps); //normally it would be -lr_unbias, but we want to maximize
										 //weight decay
			*d -= self.lr * self.lambda * *p;
		}

		//return
		delta
	}

	/// Retrieve the timestep (to allow computing manual learning rate decay)
	fn get_t(&self) -> usize {
		self.t
	}
}

/// Lookahead optimizer on top of other optimizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lookahead<Opt: Optimizer> {
	/// Sub-optimizer.
	subopt: Opt,
	/// Outer step size.
	alpha: Float,
	/// Number of taken timesteps.
	t: usize,
	/// Number of steps between paramter synchronizations.
	k: usize,
	/// Temporary storage of parameters for the k steps.
	paramssave: Vec<Float>,
}

impl<Opt: Optimizer> Lookahead<Opt> {
	/// Create new Lookahead optimizer instance using default hyperparameters
	/// (alpha = 0.5, k = 5)
	#[must_use]
	pub fn new(opt: Opt) -> Lookahead<Opt> {
		Lookahead { subopt: opt, alpha: 0.5, t: 0, k: 5, paramssave: Vec::new() }
	}

	/// Set outer step size
	pub fn set_alpha(&mut self, step: Float) -> &mut Self {
		if step <= 0.0 {
			panic!("Step size must be greater than zero!");
		}
		self.alpha = step;

		self
	}

	/// Set synchronization frequency.
	pub fn set_k(&mut self, syncfreq: usize) -> &mut Self {
		if syncfreq < 1 {
			panic!("Synchronization frequency in Lookahead must be at least k=1");
		}
		self.k = syncfreq;

		self
	}

	/// Get inner optimizer.
	pub fn get_opt(&self) -> &Opt {
		&self.subopt
	}

	/// Get inner optimizer mutably.
	pub fn get_opt_mut(&mut self) -> &mut Opt {
		&mut self.subopt
	}
}

impl<Opt: Optimizer + Serialize + DeserializeOwned> Lookahead<Opt> {
	/// Encodes the optimizer as a JSON string.
	#[must_use]
	pub fn to_json(&self) -> String {
		serde_json::to_string(self).expect("Encoding JSON failed!")
	}

	/// Builds a new optimizer from a JSON string.
	#[must_use]
	pub fn from_json(encoded: &str) -> Lookahead<Opt> {
		serde_json::from_str(encoded).expect("Decoding JSON failed!")
	}

	/// Saves the model to a file
	pub fn save(&self, file: &str) -> Result<(), std::io::Error> {
		let mut file = File::create(file)?;
		let json = self.to_json();
		file.write_all(json.as_bytes())?;
		Ok(())
	}

	/// Creates a model from a previously saved file
	pub fn load(file: &str) -> Result<Lookahead<Opt>, std::io::Error> {
		let json = fs::read_to_string(file)?;
		Ok(Lookahead::<Opt>::from_json(&json))
	}
}

impl<Opt: Optimizer> Optimizer for Lookahead<Opt> {
	/// Compute delta update from params and gradient
	fn get_delta(&mut self, params: &[Float], grad: &[Float]) -> Vec<Float> {
		//save initial parameters on start
		if self.t == 0 {
			self.paramssave = params.to_vec();
		}

		//inner update
		let mut delta = self.subopt.get_delta(params, grad);

		//timestep
		self.t += 1;

		//outer update
		if self.t % self.k == 0 {
			for ((ps, p), d) in self.paramssave.iter_mut().zip(params.iter()).zip(delta.iter_mut())
			{
				let diff = (*p + *d) - *ps; //difference between initial params and explored params
				let new = self.alpha.mul_add(diff, *ps); //outer update target params
				*d = new - *p; //calculate delta to get from current params to new params
				*ps = new; //update paramssave
			}
		}

		//return
		delta
	}

	/// Retrieve the timestep (to allow computing manual learning rate decay)
	fn get_t(&self) -> usize {
		self.t
	}
}
