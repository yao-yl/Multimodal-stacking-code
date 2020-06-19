functions {
	/**
		* Returns linear predictor for restricted Boltzman machine (RBM).
	* Assumes one-hidden layer with logistic sigmoid activation.
	*
		* @param x Predictors (N x M)
	* @param alpha First-layer weights (M x J)
	* @param beta Second-layer weights (J x (K - 1))
	* @return Linear predictor for output layer of RBM.
	*/
		vector rbm(matrix x, matrix alpha, vector beta) {
			return (tanh(x * alpha) * beta);
		}
}
data {
	int<lower=0> N;               // num train instances
	int<lower=0> M;               // num train predictors
	matrix[N, M] x;               // train predictors
	int<lower=0, upper=1> y[N];   // train category
	int<lower=1> J;               // num hidden units
	int<lower=0> Nt;              // num test instances
	matrix[Nt, M] xt;             // test predictors
	int<lower=0, upper=1> yt[Nt];  // test category
}
parameters {
	matrix[M, J] alpha;
	ordered[J] beta;
	real phi;
}
model {
	vector[N] v = rbm(x, alpha, beta);

  // priors
  to_vector(alpha) ~ normal(0, 3);
  beta ~ normal(0, 3);
  phi ~ normal(0, 3);
  // likelihood
  for (n in 1:N)
    y[n] ~ bernoulli_logit(v[n]+ phi);
}


generated quantities {
  real log_lik[N];   // train log likelihood
  real log_lik_test[Nt];   // train log likelihood

vector[ N] v = rbm(x, alpha, beta);
vector[Nt] vt= rbm(xt, alpha, beta);
  for (n in 1:N) 
     log_lik[n] = bernoulli_logit_lpmf(y[n] | v[ n]+ phi);
  for (n in 1:Nt) {
    log_lik_test[n] = bernoulli_logit_lpmf(yt[n] | vt[ n]+ phi);
  }
}
