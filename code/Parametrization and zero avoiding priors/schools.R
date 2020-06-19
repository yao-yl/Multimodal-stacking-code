### 1. Generate a set of data  ######
BB_vec=sort(c(seq(0.01, 0.5,length.out = 3),c(1), c(5,10, 50)))
sigma_vec=c(0.01,0.1,1,10,100)
N_vec=c(30,100,300)
b_vec=c(0.1,1,5)
I1=length(BB_vec)
I2=length(sigma_vec)
I3=length(N_vec)
I4=length(b_vec)

### 2. We are running the follwoing code on a cluster  ######
### For local runnning, set  arrayid = some number betweeen 1 to prod (I1, I2, I3, I4)  ######

args <-  Sys.getenv("SLURM_ARRAY_TASK_ID")
print(Sys.getenv("SLURM_ARRAY_JOB_ID"))
print(args)
arrayid <- as.integer(args[1])
set.seed(as.integer(arrayid))

library(rstan)
library(loo)
setindex=arrayInd(arrayid, .dim=c(I1, I2, I3, I4))
BB=BB_vec[setindex[1]]
sigma=sigma_vec[setindex[2]]
N=N_vec[setindex[3]] #number of  groups, J in our paper.
b=b_vec[setindex[4]]


J=20   #number of units in each group, denoted by ``N'' in our paper
mu=0   # group mean
sigma=0.01  # individual level sd  
a=alpha=beta=0.1
n.chain=8

elpd_score=test_score=rep(NA,5)
F_stat=NA
st_weight_record=rep(NA,2)
set.seed(485735)
tau=0.0001
theta=rnorm(N,mu,sd=tau)+rt(N,1)*BB  # latent group variables
y=matrix(NA, N, J)   # observed units
y_test=matrix(NA, N, J_test)   # observed units
for(i in 1:N){
  y[i,]=rnorm(J,theta[i], sd=sigma)
  y_test[i,]=rnorm(J_test,theta[i], sd=sigma)
}
group_mean=rowMeans(y)
global_mean=mean(y)
S_b=sum(  (group_mean-global_mean)^2)*J
S_w=0
for(i in 1:N)
  for (j in 1:J)
    S_w=S_w+(y[i,j]-group_mean[i])^2
F_stat=(S_b/(N-1))  /  (S_w/ (N*(J-1)))
iter=1000
stan_fit_zero=stan(file="random_effect_zero.stan", data =  list(alpha=alpha, beta=beta, N=N, J=J, y=y, J_test=J_test, y_test=y_test), chains = 1, iter=1000) 
stan_fit_cp=stan(file="random_effect.stan", data =  list(a=a, b=b, alpha=alpha, beta=beta, N=N, J=J, y=y,J_test=J_test, y_test=y_test), chains = n.chain, iter=1000, seed = 100) 
stan_fit_ncp=stan(file="random_effect_ncp.stan", data = list(a=a, b=b, alpha=alpha, beta=beta, N=N, J=J, y=y, J_test=J_test, y_test=y_test), chains = n.chain, iter=1000, seed = 100) 
sample_time=rep(NA,2*n.chain+2 )
sample_time[1]=get_elapsed_time(stan_fit_zero)[,2]
sample_time[1:n.chain+1]=get_elapsed_time(stan_fit_cp)[,2]
sample_time[1:n.chain+n.chain+1]= get_elapsed_time(stan_fit_ncp)[,2]
log_lik1=extract(stan_fit_zero, pars='log_lik', permuted=FALSE)[,1,]
log_lik2=extract(stan_fit_cp, pars='log_lik', permuted=FALSE)
log_lik3=extract(stan_fit_ncp, pars='log_lik', permuted=FALSE)
log_lik1_test=extract(stan_fit_zero, pars='log_lik_test', permuted=FALSE)[,1,]
log_lik2_test=extract(stan_fit_cp, pars='log_lik_test', permuted=FALSE)
log_lik3_test=extract(stan_fit_ncp, pars='log_lik_test', permuted=FALSE)
sim_array=array(NA, c(iter/2, 2*n.chain+1, 2))
sim_array[,1,1:2]=extract(stan_fit_zero,  permuted=FALSE, pars=c("mu","sigma"))
sim_array[,1+1:n.chain,]=extract(stan_fit_cp,  permuted=FALSE, pars=c("mu","sigma"))
sim_array[,1+n.chain+1:n.chain,]=extract(stan_fit_ncp,  permuted=FALSE, pars=c("mu","sigma"))
n_eff=matrix(NA,  2*n.chain+1, 2)
for(i in 1:(2*n.chain+1))
{
  if(i==1)
    n_eff[i,1:2]= apply(sim_array[,i,1:2], 2, effectiveSize)
  else
    n_eff[i,]= apply(sim_array[,i,], 2, effectiveSize)
}
n_eff_overall_stan=rep(NA,3)
n_eff_overall_stan[1]=1/mean( 1/summary(stan_fit_zero)$summary[1:2, "n_eff"] )
n_eff_overall_stan[2]=1/mean(  1/summary(stan_fit_cp)$summary[1:(N+3), "n_eff"] )
n_eff_overall_stan[3]=1/mean(  1/summary(stan_fit_ncp)$summary[1:(N+3), "n_eff"] )
n_eff_overall=rep(NA,5)
n_eff_overall[1]=1/mean( 1/  n_eff [1,]   )
n_eff_overall[2]=1/mean( 1/  n_eff [1+1:n.chain,]   )
n_eff_overall[3]=1/mean( 1/  n_eff [1+n.chain+1:n.chain,]   )
loo_complete_pooling=loo(stan_fit_zero)
elpd_score[1]=loo_complete_pooling$estimates[1,1] 
elpd_score[2]=loo(stan_fit_cp)$estimates[1,1]
elpd_score[3]=loo(stan_fit_ncp)$estimates[1,1]
test_score[1]=mean(log_lik1_test, na.rm = T) 
test_score[2]=mean(log_lik2_test, na.rm = T) 
test_score[3]=mean(log_lik3_test, na.rm = T) 
loo_elpd=matrix(NA, J*N ,2*n.chain+1)
test_elpd=matrix(NA, J_test*N ,2*n.chain+1)
test_elpd[,1]=apply( log_lik1_test, 2,mean)
loo_elpd[,1]=loo_complete_pooling$pointwise[,1]
for(i in 1:n.chain)
{
  loo_item=loo(log_lik2[,i,])
  loo_elpd[,1+i]=loo_item$pointwise[,1]
  test_elpd[,1+i]=apply( log_lik2_test[,i,], 2,mean)
}
for(i in 1:n.chain)
{
  loo_item=loo(log_lik3[,i,])
  loo_elpd[,n.chain+1+i]=loo_item$pointwise[,1]
  test_elpd[,n.chain+1+i]=apply( log_lik3_test[,i,], 2,mean)
}
opt=stack_with_na( lpd_point=loo_elpd[, c(1, 1+1:n.chain )],
                   lpd_point_test=test_elpd[, c(1, 1+1:n.chain )])
st_weight_record[1]=opt$full_weight[1]
elpd_score[4]=opt$loo_score
test_score[4]=opt$test_score/N/J_test
n_eff_overall[4]=  1/mean(c( 1/stacked_effective_sample_size(n_eff[c(1, 1+1:n.chain), 1], opt$full_weight), 1/stacked_effective_sample_size(n_eff[c(1, 1+1:n.chain), 2], opt$full_weight)))
opt=stack_with_na( loo_elpd[, c(1, n.chain+1+1:n.chain )], 
                   lpd_point_test=test_elpd[, c(1, n.chain+1+1:n.chain )]  )
st_weight_record[2]=opt$full_weight[1]
elpd_score[5]=opt$loo_score
test_score[5]=opt$test_score/N/J_test
n_eff_overall[5]=  1/mean(c( 1/stacked_effective_sample_size(n_eff[c(1, n.chain+1+1:n.chain), 1], opt$full_weight), 1/stacked_effective_sample_size(n_eff[c(1,n.chain+1+1:n.chain), 2], opt$full_weight)))
save(list=c("elpd_score","st_weight_record", "F_stat", "n_eff_overall", "n_eff", "n_eff_overall_stan","sample_time"),file=paste("arg_", arrayid, ".RData", sep=""))
