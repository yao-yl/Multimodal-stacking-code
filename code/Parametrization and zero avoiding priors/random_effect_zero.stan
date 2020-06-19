data{
  real alpha;
  real beta;
  int N;
  int J;
  matrix[N,J] y;
  int J_test;
  matrix[N,J_test] y_test;
  
}
parameters{
  real mu;
  real<lower=0> sigma;
}
model{
  for(i in 1:N)
    for(j in 1:J)
       y[i,j]~normal(mu, sigma);
  target += inv_gamma_lpdf(sigma^2 | alpha, beta)+ log(sigma); // jacobian
}

generated quantities{
  matrix[N,J] log_lik;
  matrix[N,J_test] log_lik_test;

  for(i in 1:N)
    for(j in 1:J)
       log_lik[i,j]=normal_lpdf(y[i,j]|mu, sigma);
       
  for(i in 1:N)
    for(j in 1:J_test)
       log_lik_test[i,j]=normal_lpdf(y_test[i,j]|mu, sigma);     
}

