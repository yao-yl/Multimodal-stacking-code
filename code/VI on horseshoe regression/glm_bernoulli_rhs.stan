
data {
  int<lower=0> n;				      // number of observations
  int<lower=0> d;             // number of predictors
  int<lower=0> n_test;				      // number of observations
  int<lower=0,upper=1> y[n];	// outputs
  matrix[n,d] x;				      // inputs
  int<lower=0,upper=1> y_test[n_test];	// outputs
  matrix[n_test,d] x_test;				      // inputs
  real<lower=0> scale_icept;	// prior std for the intercept
  real<lower=0> scale_global;	// scale for the half-t prior for tau
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
}

parameters {
  real beta0; // intercept
  vector[d] z; // auxiliary parameter
  real<lower=0> tau;			// global shrinkage parameter
  vector<lower=0>[d] lambda;	// local shrinkage parameter
  real<lower=0> caux; // auxiliary
}

transformed parameters {
  
  real<lower=0> c;
  vector[d] beta;				// regression coefficients
  vector[n] f;				// latent values
  vector<lower=0>[d] lambda_tilde;
  
  c = slab_scale * sqrt(caux);
  lambda_tilde = sqrt( c^2 * square(lambda) ./ (c^2 + tau^2* square(lambda)) );
  beta = z .* lambda_tilde*tau;
  f = beta0 + x*beta;
}

model {
  
  z ~ normal(0,1);
  lambda ~ cauchy(0,1);
  tau ~ cauchy(0, scale_global);
  caux ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
  
  beta0 ~ normal(0,scale_icept);
  y ~ bernoulli_logit(f);
}

generated quantities {
  // compute log-likelihoods for loo
  vector[n] log_lik;
  vector[n_test] log_lik_test;
  
  {
  vector[n_test] f_test;
  f_test = beta0 + x_test*beta;
  for (i in 1:n)
    log_lik[i] = bernoulli_logit_lpmf(y[i] | f[i]);
  for (i in 1:n_test)
    log_lik_test[i] = bernoulli_logit_lpmf(y_test[i] | f_test[i]);
  }
}



