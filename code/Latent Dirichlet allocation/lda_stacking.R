############################################
### Data and code for LDA examples##########
############################################

#=========================================
##data and preprocess
remotes::install_github("MansMeg/posteriordb", subdir = "rpackage")
library(posteriordb) 
ghpdb <- pdb_github()
po <- posterior("prideprejustice_paragraph-ldaK5", ghpdb)
sd <- stan_data(po)
sc <- stan_code(po)
set.seed(100) 
train_sample=sort(sample(1:sd$N, round(0.7*sd$N)))
w_train=sd$w[train_sample]
w_test=sd$w[-train_sample]
doc_train=sd$doc[train_sample]
doc_test=sd$doc[-train_sample]
N=length(w_train)
N_test=length(w_test)
sd_train=sd
sd_train$w=w_train
sd_train$w_test=w_test
sd_train$N=N
sd_train$doc=doc_train
sd_train$doc_test=doc_test
sd_train$N_test=N_test
sd_train$K=as.integer(5) ## varies in the experiment from 3 to 15.
save(sd_train, file="sd_train.RData")
# =============================================================
# stan model, save as a seperate file lda_test.stan
# data {
# 	int<lower=2> V;               // num words
# 	int<lower=1> M;               // num docs
# 	int<lower=1> N;               // total word instances
# 	int<lower=1> N_test;                
# 	int<lower=1> K;               // num topics
# 	int<lower=1,upper=V> w[N];    // word n
# 	int<lower=1,upper=V> w_test[N_test];    // word in test set
# 	int<lower=1,upper=M> doc[N];  // doc ID for word n
# 	int<lower=1,upper=M> doc_test[N_test];  // doc ID for test words
# 	vector<lower=0>[K] alpha;     // topic prior
# 	vector<lower=0>[V] beta;      // word prior
# }
# parameters {
# 	simplex[K] theta[M];   // topic dist for doc m
# 	simplex[V] phi[K];     // word dist for topic k
# }
# model {
# 	for (m in 1:M)
# 		theta[m] ~ dirichlet(alpha);  // prior
# 	for (k in 1:K)
# 		phi[k] ~ dirichlet(beta);     // prior
# 	for (n in 1:N) {
# 		real gamma[K];
# 		for (k in 1:K)
# 			gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
# 		target += log_sum_exp(gamma);  // likelihood;
# 	}
# } 
# generated quantities{
# 	real log_lik[N];
# 	real log_lik_test[N_test];
# 	for (n in 1:N) {
# 		real temp[K];
# 		for (k in 1:K)
# 			temp[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
# 		log_lik[n] = log_sum_exp(temp);  
# 	}
# 	for (n in 1:N_test) {
# 		real temp[K];
# 		for (k in 1:K)
# 			temp[k] = log(theta[doc_test[n], k]) + log(phi[k, w_test[n]]);
# 		log_lik_test[n] = log_sum_exp(temp);  
# 	}
# }

## ======================================================
## running on cluster
## do not run
args <-  Sys.getenv("SLURM_ARRAY_TASK_ID")
print(Sys.getenv("SLURM_ARRAY_JOB_ID"))
print(args)
arrayid <- as.integer(args[1])
set.seed(as.integer(arrayid))
library(rstan)
lda_fit=stan(file="lda_test.stan", data=sd_train, iter=2000, thin =2, seed=arrayid, chains = 1)
log_lik_iter=extract(lda_fit, pars="log_lik", permuted=F)[,1,]
log_lik_test_iter=extract(lda_fit, pars="log_lik_test", permuted=F)[,1,]
save(log_lik_iter,log_lik_test_iter,file=paste("arg_", arrayid, ".RData", sep=""))
rm(list=ls())

## ======================================================
## extract log likelihood, predictive density on test data
## and loo from the cluster

st_weight_list=list()
stacking_loo_result_list=list()
stacking_loo_result_vec=c()
stacking_loo_test_vec=c()
stacking_loo_ind_list=list()
log_lik_test_ind_list=list()
K_vec=c()


num_cluster_vec=c(5, 3,4,7,10,15)  ## the setting of LDA number of clusters
process.index=1  ## we have 6 

for(process.index in 1:5){

num_cluster=num_cluster_vec[process.index]
name_char=c("05","03","04","07","10","15")[process.index]

K_vec[process.index]=num_cluster

library(rstan)
library(loo)
K=30
sample_flag=rep(0,K)
iter=500
N=23014
N_test= 9863
log_lik_mat=array(NA, c(iter,K,N))
log_lik_test_mat=array(NA, c(iter,K,N_test))

## aggregrate log likelihood and predictive density on test data into a matrix
## choose  the  dir for  the saved running results
for(i in 1:K){
	file_name=paste("~/Desktop/lda_data/lda4000/lda",name_char, "/arg_2_", i, ".RData", sep="") 
	if(file.exists(file_name))  {
		load(file_name)
		sample_flag[i]=1
		log_lik_mat[,i,]=log_lik_iter
		log_lik_test_mat[,i,]=log_lik_test_iter
		rm(list = c("log_lik_iter","log_lik_test_iter"))
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
}

st_weight=stacking_weights( loo_elpd)

stacking_loo_result=log_score_loo(w=st_weight, lpd_point= loo_elpd)
log_lik_test_agg=test_aggregrate(log_lik_test)
stacking_loo_test=log_score_loo(w=st_weight, lpd_point= log_lik_test_agg)

st_weight_list[[process.index]]=st_weight
stacking_loo_result_vec[process.index]=stacking_loo_result
stacking_loo_ind_list[[process.index]]=apply(loo_elpd, 2, sum )
stacking_loo_test_vec[process.index]=stacking_loo_test
log_lik_test_ind_list[[process.index]]=apply(log_lik_test_agg, 2, sum )
}


#save(st_weight_list, stacking_loo_result_vec, stacking_loo_ind_list, stacking_loo_test_vec, log_lik_test_ind_list, K_vec, file="~/Desktop/lda_final_result_4000iter.RData")

num_of_cluster=5

num_effective_chain_vec=c()
for(i in 1:num_of_cluster)
{
	num_effective_chain_vec[i]=1/sum( st_weight_list[[i]]^2 )
}

plot(K_vec, num_effective_chain_vec)
#=================================================================
## graphs

#load("lda_final_result.RData")

pdf("~/desktop/lda_result04.pdf", height=2.7,width=7)
par(mfcol=c(1,2),oma=c(1.5 ,1,1.5,0), pty='m',mar=c(1,2.4,1,2) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.8, cex.main=0.9,xpd=F)
print(range( c( range(stacking_loo_result_vec/N), range((unlist( stacking_loo_ind_list) ))/N)   ))
plot(K_vec,stacking_loo_result_vec/N, ylim=c(-6.7, -6.3) ,  xlim=c(2,11) ,axes=F,xlab="",ylab=" ", pch=19, col= alpha("darkred", alpha = 0.7), cex=0.7)
lines(K_vec[order(K_vec)] ,(stacking_loo_result_vec/N)[order(K_vec)],col=alpha("darkred", alpha = 0.7))
for(i in 1: num_of_cluster){
	n_points=length(stacking_loo_ind_list[[i]])
	points(x= rep( K_vec[i],n_points)+runif(n_points, -0.03,0.03), y=  stacking_loo_ind_list[[i]]/N, 
				 col=alpha("grey", alpha = 0.5), pch=19, cex=0.7)
}
axis(2,   las=2, lwd=0.5, at=c(-6.5,-6.7,-6.3))
axis(2,   las=2, lwd=0.5, at=c(-6.71,-6.29), hadj=1.5,labels = c("worse fit", "better fit"), tick = F)
axis(1,   padj=-1, lwd=0.5, at=K_vec)
mtext(3, text="LOO mean log\n predictive densities",line=0,cex=0.7)
mtext(1, text="number of topics",line=1,cex=0.7)
text(8, -6.38, labels = "stacking",cex=0.6, col='darkred')
text(8, -6.6, labels = "individual chains",cex=0.6, col='grey20')
box(bty='l',lwd=0.5)
print(range( c( range(stacking_loo_test_vec/N_test), range((unlist( log_lik_test_ind_list) ))/N_test)   ) )
plot(K_vec,stacking_loo_test_vec/N_test, ylim=c(-7.1, -6.6) ,  xlim=c(2,11) ,axes=F,xlab="",ylab=" ", pch=19, col= alpha("darkred", alpha = 0.7), cex=0.7)
lines(K_vec[order(K_vec)] ,(stacking_loo_test_vec/N_test)[order(K_vec)],col=alpha("darkred", alpha = 0.7))
for(i in 1: num_of_cluster){
	n_points=length(stacking_loo_ind_list[[i]])
	points(x= rep( K_vec[i],n_points)+runif(n_points, -0.03,0.03), y=  log_lik_test_ind_list[[i]]/N_test, col=alpha("grey", alpha = 0.5), pch=19, cex=0.7)
}
axis(2,   las=2, lwd=0.5, at=c(-7, -6.8,-6.6))
axis(1,   padj=-1, lwd=0.5, at=K_vec)
mtext(3, text="Test mean log\n predictive densities",line=0,cex=0.7)
mtext(1, text="number of topics",line=1,cex=0.7)
box(bty='l',lwd=0.5)
text(8, -6.65, labels = "stacking",cex=0.6, col='darkred')
text(9, -7, labels = "individual chains",cex=0.6, col='grey20')
dev.off()








