### Replication  code for stacking in horseshoe regression ##
###  1. Running VI 200 times with different initialiation ##
#We run this on a cluster with arrayid=1:200
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set.seed(as.integer(arrayid))
library(rstan)
datafile <- 'leukemia.RData'
load(datafile,verbose=T)
x <- scale(x)
d <- NCOL(x)
n <- NROW(x)
# compile the model
stanmodel <- stan_model(file='glm_bernoulli_rhs.stan')
scale_icept=10
slab_scale=5
slab_df=4
# data and prior
tau0 <- 1/(d-1) * 2/sqrt(n) # should be a reasonable scale for tau (see Piironen&Vehtari 2017, EJS paper)
scale_global=tau0
data <- list(n=n, d=d, x=x, y=as.vector(y), scale_icept=10, scale_global=tau0,
						 slab_scale=5, slab_df=4)
# NUTS solution (increase the number of iterations, here only 100 iterations to make this script run relatively fast)
fit_nuts <- stan(file="glm_bernoulli_rhs1.stan", data=data, iter=1200, control=list(adapt_delta=0.9))
# ADVI
fit_advi <- vb(stanmodel, data=data,iter=5e6,output_samples=2e3,tol_rel_obj=0.005,eta = 1, seed=as.integer(arrayid))
save(fit_advi,file=paste("arg_log_lik/arg_", arrayid, ".RData", sep=""))


## 2. Collect sampled objects ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
library(rstan)
library(loo)
sample_flag=rep(0,200)
log_lik=array(NA, c(1000,200,72))
for(i in 1:200){
	file_name=paste("arg_log_lik/loglik_", i, ".RData", sep="")
	if(file.exists(file_name))  {
		load(file_name)
		if   ( (dim(SS[,1,] )[1]==1000 ) & (sum(is.na(SS[,1,] ))==0  )& (sum(is.nan(SS[,1,] ))==0  ))   {
			sample_flag[i]=1
			log_lik[,i,]=SS[,1,] 
		}
		rm(list=c("SS"))
	}
}

S=sum(sample_flag)
log_lik=log_lik[,sample_flag==1,]
elpd_chain=ploo=c()
loo_list=list()
for(i in 1:S){
	log_lik_matrix=log_lik[,i,]
	loo_chain=loo(log_lik_matrix)
	loo_list[[i]]=loo_chain
}
n_sample=72
loo_elpd=matrix(NA,n_sample, S)
for(i in 1:S){
	loo_item=loo_list[[i]]
	loo_elpd[,i]=loo_item$pointwise[,1]
}
new_flag=rep(0,S)
for( i in 1:S)
	if(  sum( is.na( loo_elpd[, i])) ==0)
		new_flag[i]=1
S_nz=sum(new_flag)
loo_elpd=loo_elpd[,which(new_flag==1)]
print(dim(loo_elpd))


###### 3. obtain weight for each run ######
st_weight=stacking_weights( lpd_point=loo_elpd, optim_control=list(maxit=3000, abstol=1e-6))
bma_weight=pseudobma_weights(lpd_point=loo_elpd, BB=FALSE)
full_weight=rep(0,S)
full_weight[which(new_flag==1)]=st_weight$wts
full_weight_bma=rep(0,S)
full_weight_bma[which(new_flag==1)]=bma_weight

 