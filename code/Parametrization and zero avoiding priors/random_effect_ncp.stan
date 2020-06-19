data{
  real a;
  real b;
  real alpha;
  real beta;
  int N;
  int J;
  matrix[N,J] y;
  int J_test;
  matrix[N,J_test] y_test;
}
parameters{
  vector[N] theta_trans;
  real mu;
  real<lower=0> tau;
  real<lower=0> sigma;
}
transformed parameters{
  vector[N] theta;
  theta=mu+ tau*theta_trans;
}

model{
  for(i in 1:N)
    for(j in 1:J)
       y[i,j]~normal(theta[i], sigma);
  theta_trans~normal(0,1);
  target += inv_gamma_lpdf(tau^2 | a, b) + log(tau);
  target += inv_gamma_lpdf(sigma^2 | alpha, beta)+ log(sigma);
}


generated quantities{
  matrix[N,J] log_lik;
  matrix[N,J_test] log_lik_test;

  for(i in 1:N)
    for(j in 1:J)
       log_lik[i,j]=normal_lpdf(y[i,j]|theta[i], sigma);
       
  for(i in 1:N)
    for(j in 1:J_test)
       log_lik_test[i,j]=normal_lpdf(y_test[i,j]|theta[i], sigma);     
}
