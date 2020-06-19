#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# replication code for neural net MNIST stacking
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

setwd("~")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# stan code ##############################
# adapted from Bob Carpenter's Stan code  
# https://github.com/stan-dev/example-models/blob/master/knitr/neural-nets/nn-simple.stan
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#stan code, save it as a seperate file named 2_class_nn.stan

# functions {
# 	/**
# 		* Returns linear predictor for restricted Boltzman machine (RBM).
# 	* Assumes one-hidden layer with logistic sigmoid activation.
# 	*
# 		* @param x Predictors (N x M)
# 	* @param alpha First-layer weights (M x J)
# 	* @param beta Second-layer weights (J x (K - 1))
# 	* @return Linear predictor for output layer of RBM.
# 	*/
# 		vector rbm(matrix x, matrix alpha, vector beta) {
# 			return (tanh(x * alpha) * beta);
# 		}
# }
# data {
# 	int<lower=0> N;               // num train instances
# 	int<lower=0> M;               // num train predictors
# 	matrix[N, M] x;               // train predictors
# 	int<lower=0, upper=1> y[N];   // train category
# 	int<lower=1> J;               // num hidden units
# 	int<lower=0> Nt;              // num test instances
# 	matrix[Nt, M] xt;             // test predictors
# 	int<lower=0, upper=1> yt[Nt];  // test category
# }
# parameters {
# 	matrix[M, J] alpha;
# 	ordered[J] beta;
# 	real phi;
# }
# model {
# 	vector[N] v = rbm(x, alpha, beta);
# 	
# 	// priors
# 	to_vector(alpha) ~ normal(0, 3);
# 	beta ~ normal(0, 3);
# 	phi ~ normal(0, 3);
# 	// likelihood
# 	for (n in 1:N)
# 		y[n] ~ bernoulli_logit(v[n]+ phi);
# }
# 
# 
# generated quantities {
# 	real log_lik[N];   // train log likelihood
# 	real log_lik_test[Nt];   // train log likelihood
# 	
# 	vector[ N] v = rbm(x, alpha, beta);
# 	vector[Nt] vt= rbm(xt, alpha, beta);
# 	for (n in 1:N) 
# 		log_lik[n] = bernoulli_logit_lpmf(y[n] | v[ n]+ phi);
# 	for (n in 1:Nt) {
# 		log_lik_test[n] = bernoulli_logit_lpmf(yt[n] | vt[ n]+ phi);
# 	}
# }
# 
# 




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# read data ##############################
# do NOT run. We have saved the data in input.RData
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

source('read-mnist.R');
load_mnist();
## the data can be downlaoded fom https://github.com/stan-dev/example-models/tree/master/knitr/neural-nets/mnist

standardize <- function(u) (u - mean(u)) / sd(u);
yp1 <- train$y + 1;
K <- max(yp1);
N <- length(yp1);
x_std <- train$x;
M <- dim(x_std)[2];
J <- 50;
for (k in 1:K) {
	if (sum(x_std[ , k] != 0) > 1)
		x_std[ , k] <- standardize(x_std[ , k]);
}

xt_std <- test$x;
for (k in 1:K) {
	if (sum(x_std[ , k] != 0) > 1)
		xt_std[ , k] <- (xt_std[ , k] - mean(x_std[ , k])) / sd(x_std[ , k]);
}
ytp1 <- test$y + 1;
Nt <- dim(xt_std)[1];
test_index =   which(test$y ==1| test$y ==2)
N_MAX = 10000
train_index =   c(which(train$y ==1)[1:5000], which(train$y ==2)[1:5000])
library(rstan);
nn_model <- stan_model("2_class_nn.stan");
Nt_MAX = length(test_index);
dim(x_std)
mnist_data <- list(K = K, J = J, M=dim(x_std)[2],
									 x = x_std[train_index, ], N = N_MAX, y = yp1[train_index]-2,
									 xt = xt_std[test_index, ], Nt = Nt_MAX, yt = ytp1[test_index]-2);
# save(mnist_data, file="input10000.RData")
# load(file="input.RData")
nn_model <- stan_model("2_class_nn.stan");
fits <- sampling(nn_model,data = mnist_data, iter=2, chains=1, refresh=1);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# inference #####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# cluster setting,  We are running 50 paralle chains on Columbia shared cluster
# set args=1 on local machines.
args <-  Sys.getenv("SLURM_ARRAY_TASK_ID")
print(Sys.getenv("SLURM_ARRAY_JOB_ID"))
print(args)
arrayid <- as.integer(args[1])
set.seed(as.integer(arrayid))

library(rstan)
load(file="input.RData")
fits=stan(file="2_class_nn.stan", data=mnist_data, iter=2000, thin =2, init_r =10 , seed=arrayid, chains = 1)
log_lik_iter=extract(fits, pars="log_lik", permuted=F, inc_warmup=T)[,1,]
log_lik_test_iter=extract(fits, pars="log_lik_test", permuted=F, inc_warmup=T)[,1,]
save(log_lik_iter,log_lik_test_iter,file=paste("arg_", arrayid, ".RData", sep=""))


#init_r =10 controls how dispersed the initialization is. We choose  init_r =20 for J=10 in over-dispersed settings.



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# aggregrations #######
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

library(loo)
K=50 #choose how many paralles runs you have saved
sample_flag=rep(0,K)
iter=500
N=mnist_data$N
N_test= mnist_data$Nt
log_lik_mat=array(NA, c(iter,K,N))
log_lik_test_mat=array(NA, c(iter,K,N_test))

## aggregrate log likelihood and predictive density on test data into a matrix
## choose  the  dir for  the saved running results
for(i in 1:K){
	file_name=paste("loglik/arg_", i, ".RData", sep="") 
	if(file.exists(file_name))  {
		load(file_name)
		sample_flag[i]=1
		log_lik_mat[,i,]=log_lik_iter[501:1000,]
		log_lik_test_mat[,i,]=log_lik_test_iter[501:1000,]
	}
}

## discard dead nodes (running maximum 24 hours for 2000 iters/or 48 hour for 4000 iters)
S=sum(sample_flag)
log_lik=log_lik_mat[,sample_flag==1,]
log_lik_test=log_lik_test_mat[,sample_flag==1,]

elpd_chain=ploo=c()
loo_list=list()
for(i in 1:S){
	log_lik_matrix=log_lik[,i,]
	loo_chain=loo(log_lik_matrix)
	loo_list[[i]]=loo_chain
}
n_sample=N
loo_elpd=matrix(NA,n_sample, S)
for(i in 1:S){
	loo_item=loo_list[[i]]
	loo_elpd[,i]=loo_item$pointwise[,1]
	loo_elpd[is.na(loo_elpd[,i]), i]=min(loo_elpd[,i], na.rm = T)
}
st_weight=stacking_weights(loo_elpd, lambda=1.0001)
stacking_loo_result=log_score_loo(w=st_weight, lpd_point= loo_elpd) 
log_lik_test_agg=test_aggregrate(log_lik_test)
stacking_loo_test=log_score_loo(w=st_weight, lpd_point= log_lik_test_agg)
# stacking_loo_test is the test elpd of stacking


