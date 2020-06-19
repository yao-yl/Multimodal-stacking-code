functions {
  vector gp_pred_t_rng(real[] x2, 
                     vector f1, real[] x1,
                     real alpha, real rho,  real delta) {
    int N1 = rows(f1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho) + diag_matrix(rep_vector(delta, N1));
      matrix[N1, N1] L_K = cholesky_decompose(K);
      vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, f1);
      vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
      matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho);
      vector[N2] f2_mu = (k_x1_x2' * K_div_y1);
      matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      matrix[N2, N2] cov_f2 = cov_exp_quad(x2, alpha, rho) - v_pred' * v_pred
                              + diag_matrix(rep_vector(delta, N2));
      f2 = multi_normal_rng(f2_mu, cov_f2);
    }
    return f2;
  }
}

data {
  int<lower=1> N;
  real x[N];
  vector[N] y;
  
  int<lower=1> N_test;
  real x_test[N_test];
  real y_test[N_test];

}

transformed data {
  real delta = 1e-10;
}

parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
  vector[N] eta;
}

transformed parameters{
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x, alpha, rho)+ diag_matrix(rep_vector(delta, N));
    
    L_K = cholesky_decompose(K);
    f = L_K * eta;
  }
}
model {
  rho ~ cauchy(0,3);
  alpha ~ cauchy(0,3);
  sigma ~ cauchy(0,3);
  eta ~ std_normal();
  y ~ student_t(2,f, sigma);
}


generated quantities{
  vector[N]  log_lik;
  vector[N_test] f_predict = gp_pred_t_rng(x_test, f, x, alpha, rho,  delta);
  vector[N_test] log_lik_test;
  
  for (i in 1:N)
  log_lik[i] = student_t_lpdf (y[i]|2,f[i], sigma);
  for (i in 1:N_test)
  log_lik_test[i] = student_t_lpdf(y_test[i]|2, f_predict[i], sigma);
}
