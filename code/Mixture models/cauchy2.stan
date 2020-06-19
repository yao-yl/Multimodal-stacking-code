data {
  int n;
  vector[n] y;
}
parameters {
   real mu;
}
model {
  y ~ cauchy(mu, 1); 
}

generated quantities {   
   vector[n] log_lik;
for (i in 1:n)
log_lik[i] = cauchy_lpdf(y[i]| mu, 1);
}

