data {
  int<lower=2> K;               // num topics
  int<lower=2> V;               // num words
  int<lower=1> M;               // num docs
  int<lower=1> N;               // total word instances
  int<lower=1,upper=V> w[N];    // word n
  int<lower=1,upper=M> doc[N];  // doc ID for word n
  vector<lower=0>[K] alpha;     // topic prior
  vector<lower=0>[V] beta;      // word prior
}
parameters {
  positive_ordered[K] theta_first;
  simplex[K] theta_ex_first[M-1];   // topic dist for doc m
  simplex[V] phi[K];     // word dist for topic k
}
transformed parameters {
  simplex[K] theta_first_transform = theta_first / sum(theta_first);
  simplex[K] theta[M];
  theta[1]=theta_first_transform;
  theta[2:M]=theta_ex_first;
}
model {
  for(k in 1:K)
     theta_first[k]~gamma(alpha[k], 1);
  for (m in 1:(M-1))
    theta_ex_first[m] ~ dirichlet(alpha);  // prior
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);     // prior
  for (n in 1:N) {
    real gamma[K];
    for (k in 1:K)
      gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
    target += log_sum_exp(gamma);  // likelihood;
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
     real gamma[K];
     for (k in 1:K)
       gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
     log_lik[n] = log_sum_exp(gamma); 
   }
}
