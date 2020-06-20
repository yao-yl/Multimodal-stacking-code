log_mean_exp=function(v)
{
	max(v)+ log(mean(exp(v- max(v) )))
}

log_sum_exp=function(v)
{
	max(v)+ log(sum(exp(v- max(v) )))
}

log_score_loo <- function(w, lpd_point) {
	N=dim(lpd_point)[1]
	weight_log_sum_exp<- function(v)
	{
		return(max(v)+ log( exp(v- max(v))%*% w    ))  
	}
	return (sum(apply(lpd_point, 1,  weight_log_sum_exp )) )
}


stacking_opt_stan='
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N,K] lpd_point;
  vector[K] lambda;
}
transformed data{
  matrix[N,K] exp_lpd_point; 
  exp_lpd_point=exp(lpd_point);
}
parameters {
   simplex[K] w;
}
transformed parameters{
  vector[K] w_vec;
  w_vec=w;
}
model {
  for (i in 1: N) {
    target += log( exp_lpd_point[i,] * w_vec );
  }
  w~dirichlet(lambda);
}
'

cat("First time compiling may take one minute")
stan_model_object=stan_model(model_code=stacking_opt_stan)
stacking_weights=function(lpd_point, lambda=1.0001, stan_model_object=stan_model_object, stack_iter=100000)
{
	K=dim(lpd_point)[2]
	s_w=optimizing(stan_model_object,  data = list(N=dim(lpd_point)[1], K=K, lpd_point=lpd_point, lambda=rep(lambda, dim(lpd_point)[2])), iter=stack_iter)$par[1:K] 
	return(s_w)
} 

chain_stack= function(fits,log_lik_char="log_lik",lambda=1.0001, stack_iter=100000, print_progress=TRUE  ){
	log_lik_mat=extract(fits, pars=log_lik_char, permuted=F, inc_warmup=F)
	n= dim(log_lik_mat)[3]
	K= dim(log_lik_mat)[2]
	S= dim(log_lik_mat)[1]
	if(print_progress==TRUE){
		cat(paste("Stacking", K, "chains, with",n, "data points and", S,  "posterior draws;\n using stan optimizer, max iterations =",stack_iter ,"\n ..." ))
		sysTimestamp=Sys.time()
	}
	loo_elpd=matrix(NA,n, S)
	options(warn=-1)
	loo_chain=apply(log_lik_mat, 2, function(lp){
		loo_obj=loo(lp)
		return(c(loo_obj$pointwise[,1], loo_obj$diagnostics
						 $pareto_k ))  
	}) 
	options(warn=0)
	loo_elpd= loo_chain[1:n, ]
	chain_weights=stacking_weights(lpd_point=loo_elpd, lambda=lambda, stan_model_object=stan_model_object, stack_iter=stack_iter)
	pareto_k=loo_chain[(n+1):(2*n), ]
	if(print_progress==TRUE){
	cat("done")
	cat(paste("\n Total elapsed time for approximate LOO and stacking =", round(Sys.time()
-sysTimestamp,digits=2), "s" ))
	}
	return( list(chain_weights=chain_weights, pareto_k=pareto_k ))
}
 

print_k=function(stack_obj=stack_obj){
	k=as.vector(stack_obj$pareto_k)  
	kcut <- loo:::k_cut(k)
	count <-table(kcut)
	out <- cbind(Count = count, Proportion = round( prop.table(count), digits = 3)  )
	noquote(cbind(c( "(good)", "(ok)", "(bad)", "(very bad)" ), out ))
}
 


mixture_draws= function (individual_draws,  weight, random_seed=1, S=NULL, permutation=TRUE)
{
	set.seed(random_seed)
	S_sample=dim(individual_draws)[1]
	K=dim(individual_draws)[2]
	if(is.null(S))
		S=S_sample
	if(permutation==TRUE)
		individual_draws=individual_draws[sample(1:S_sample), ]	 # random permutation of draws
	integer_part=floor(S*weight)
	existing_draws=sum(integer_part)
	if(existing_draws<S){
		remaining_draws=S-existing_draws
		update_w=(weight- integer_part/S)*  S / remaining_draws
		remaining_assignment=sample(1:K, remaining_draws, prob =update_w , replace = F)
		integer_part[remaining_assignment] =integer_part[remaining_assignment]+1
	}
	integer_part_index=c(0,cumsum(integer_part))
	mixture_vector=rep(NA, S)
	for(k in 1:K){
		if((1+integer_part_index[k])<=integer_part_index[k+1])
		mixture_vector[(1+integer_part_index[k]):integer_part_index[k+1]]=individual_draws[1:integer_part[k],k]
	}
	return(mixture_vector)
}

# An Example:
# save this as the stan inference code cauchy.stan:
# 	data {
# 		int n;
# 		vector[n] y;
# 	}
# parameters {
# 	real mu;
# }
# model {
# 	y ~ cauchy(mu, 1); 
# }
# 
# generated quantities {   
# 	vector[n] log_lik;
# 	for (i in 1:n)
# 		log_lik[i] = cauchy_lpdf(y[i]| mu, 1);
# }
# set.seed(100)
# mu=c(-10,10)
# n=100
# y=rep(NA, n)
# p=1/2
# y[1:(n*p)]=rcauchy(n*(p),mu[1], 1)
# y[(n*(p)+1):n]=rcauchy(n*(p),mu[2], 1)
# K=8
# set.seed(100)
# stan_fit=stan("cauchy.stan", data=list(n=n, y=y),chains = K , seed=100)
# mu_sample=extract(fit_sample, permuted=F, pars="mu")[,,"mu"]
# 
# stack_obj=chain_stack(stan_fit)
# chain_weights = stack_obj$chain_weights
# round(sum(chain_weights[which( apply(mu_sample,2, mean)>0) ]), digits = 3)
# resampling=mixture_draws(individual_draws=mu_sample, weight= chain_weights)
# mean(resampling>0) 
