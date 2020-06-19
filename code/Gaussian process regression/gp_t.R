#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run stacking, BMA, P-BMA, and unifrom weighting on gp regression with t likelihood. 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
library(rstan)
library(loo)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
source("stacking_utlitity.R")
m=stan_model("t_reg.stan")
##  Experiment setting------------------------------------------------------
##  Original experiment setting:
#   version  R version 3.6.1 (2019-07-05)
#   os       macOS Catalina 10.15.4      
#   system   x86_64, darwin15.6.0        
#   ui       RStudio 
#   loo      2.1.0
#   rstan    2.19.3 

##  For the graph  "gp_stacking.pdf" in the paper, we repeat all sumulations below 
##  80 times. Each time we generate a different random seed.
##  To save time, you may only run once.
### GP-t-regression on R. Neal's data. The data is generated with 
### f=0.3 + 0.4*x + 0.5 * sin(2.7*x) + 1.1/(1 + x^2 ).
### Some part of data is "outlier" with larger sigma_2,
### the remaining has a gaussian noise sigma_1.
### We fit a Gaussian process regression with t-likelihood, 
### for model details see Vanhatalo et al 2009.
### We vary sigma_2 to see 1) how the posterior change 
### and 2) how stacking help overcome non-mixing.

sigma2_vec=c(1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.25,0.2,0.15,0.1)
loo_test_max=matrix(NA, length(sigma2_vec), 6)
n=20
concentrate= 5
set.seed(20948)
x=sort(runif(n,-3,3))
f=0.3 + 0.4*x + 0.5 * sin(2.7*x) + 1.1/(1 + x^2 )
n=20
sd1=0.1
outlier_proption=0.3
id_out=sort(sample(1:n, size=round(outlier_proption*n ), prob =  exp( - (((1:n)-.4*n  )/n*concentrate )  ^2)  ))
noise1=rnorm(n,0,1)

for(sim_i in 1:length(sigma2_vec)){
	concentrate= 5
	sd2=sigma2_vec[sim_i]
	y=f+sd1 * noise1
	y[id_out]=f[id_out]+abs(   sd2* noise1[id_out]   ) * rep(c(1,-1), round(length(id_out)/2) )
	n.test= 300
	x.test=sort(runif(n.test,-3,3))
	f.test=0.3 + 0.4*x.test + 0.5 * sin(2.7*x.test) + 1.1/(1 + x.test^2 )
	id_out_test=sort(sample(1:n.test, size=round(outlier_proption*n.test ), prob =  exp( - (((1:n.test)-.4*n.test  )/n.test*concentrate )^2)  ))
	y.test=f.test+rnorm(n.test, 0,sd1)
	y.test[id_out_test]=f.test[id_out_test]+abs(  rnorm(length(id_out_test), 0,sd2)   ) * rep(c(1,-1), round(length(id_out_test)/2) )
	col.vec=rep(1, n)
	col.vec[id_out]=2
	plot(x, y, col= col.vec )
	lines(x, f, col=4)
	col_vec_test=rep(1, n.test)
	col_vec_test[id_out_test]=2
	plot(x.test, y.test, col=   col_vec_test )
	lines(x.test, f.test, col=4)
	fit <- stan(file = "~/Desktop/gp/t_reg.stan", 
							data=list(N=length(x), x=x, y=y, N_test=length(x.test), 
												x_test=x.test,y_test=y.test),
							init_r=10 , chains = 8, iter = 8000,seed=5838298,
							control=list(max_treedepth=5) )
	print(fit)
	temp=summary(fit)
	max_r=max(temp$summary[,"Rhat"])
	fit_draw_lik=extract(fit, pars="log_lik", permuted=F)
	fit_draw_lik_test=extract(fit, pars="log_lik_test", permuted=F)
	fit_draw_lp=extract(fit, pars="lp__", permuted=F)
	fit_draw_lp=fit_draw_lp[,,1]
	library(loo)
	options(mc.cores = 8)
	chains=8
	loo_elpd=matrix(NA,n, chains)
	for(i in 1:chains){
		log_lik_matrix=fit_draw_lik[,i,]
		loo_elpd[,i]=(loo(log_lik_matrix)$pointwise[,1])
	}
	#st_weight=stacking_weights( loo_elpd, lambda=1.0000)
	st_weight=stack_with_na( loo_elpd, lambda=1.00000001)$full_weight
	st_weight[is.na(st_weight)]=0
	stacking_loo_result=log_score_loo(w=st_weight, lpd_point= loo_elpd)
	loo_chain=apply(loo_elpd,  2, sum )*n.test/n
	log_lik_test_agg=test_aggregrate(fit_draw_lik_test)
	stacking_loo_test=log_score_loo(w=st_weight, lpd_point= log_lik_test_agg)
	unif_test=log_score_loo(w=rep(1/8, 8), lpd_point= log_lik_test_agg)
	ind_test = apply(log_lik_test_agg, 2, sum )
	bma_test=log_score_loo(BMA_weight(fit_draw_lp), lpd_point= log_lik_test_agg)
	pesudo_bma=log_score_loo(pesudo_BMA_weight_chain(loo_elpd), lpd_point= log_lik_test_agg)
	select_test=ind_test [which.max(loo_chain)]
	loo_test_max[sim_i,]=  c(stacking_loo_test, unif_test,bma_test, pesudo_bma,select_test, max_r )
}

print(loo_test_max)
## We expect the first column (stacking) to be the largest.

###  Likewise we can change the concentration factor.
###  With larger concentration, the outliers are more 
###  concentrated in the x-space.

concentrate_vec=c(8,7,6,5,3,2)
loo_test_max=matrix(NA, length(concentrate_vec), 6)
n=20
set.seed(20948)
x=sort(runif(n,-3,3))
f=0.3 + 0.4*x + 0.5 * sin(2.7*x) + 1.1/(1 + x^2 )
n=20
sd1=0.1
sd2=0.3
outlier_proption=0.3
for(sim_i in 1:length(concentrate_vec)){
	concentrate= concentrate_vec[sim_i]
	id_out=sort(sample(1:n, size=round(outlier_proption*n ), prob =  exp( - (((1:n)-.4*n  )/n*concentrate )  ^2)  ))
	y=f+rnorm(n, 0,sd1)
	y[id_out]=f[id_out]+abs(  rnorm(length(id_out), 0,sd2)   ) * rep(c(1,-1), round(length(id_out)/2) )
	n.test= 300
	x.test=sort(runif(n.test,-3,3))
	f.test=0.3 + 0.4*x.test + 0.5 * sin(2.7*x.test) + 1.1/(1 + x.test^2 )
	id_out_test=sort(sample(1:n.test, size=round(outlier_proption*n.test ), prob =  exp( - (((1:n.test)-.4*n.test  )/n.test*concentrate )^2)  ))
	y.test=f.test+rnorm(n.test, 0,sd1)
	y.test[id_out_test]=f.test[id_out_test]+abs(  rnorm(length(id_out_test), 0,sd2)   ) * rep(c(1,-1), round(length(id_out_test)/2) )
	col.vec=rep(1, n)
	col.vec[id_out]=2
	plot(x, y, col= col.vec )
	lines(x, f, col=4)
	col_vec_test=rep(1, n.test)
	col_vec_test[id_out_test]=2
	plot(x.test, y.test, col=   col_vec_test )
	lines(x.test, f.test, col=4)
	fit <- stan(file = "~/Desktop/gp/t_reg.stan", 
							data=list(N=length(x), x=x, y=y, N_test=length(x.test), 
												x_test=x.test,y_test=y.test),
							init_r=10 , chains = 8, iter = 8000,seed=5838299,
							control=list(max_treedepth=5) )
	print(fit)
	temp=summary(fit)
	max_r=max(temp$summary[,"Rhat"])
	fit_draw=extract(fit)
	par(mfrow=c(1,2),  mar=c(1,1,1,1))
	plot(x, y, col=   col.vec )
	lines(  x[order(x)], f[order(x)], col='blue')
	lines(  x[order(x)], colMeans (fit_draw$f)[order(x)])
	for( s in 1:1000)
		lines(x[order(x)],  fit_draw$f[s,order(x)  ], col='grey', lwd=0.2)
	col.vec_test=rep(1, n.test)
	col.vec_test[id_out_test]=2
	plot(x.test, y.test, col=   col.vec_test, pch=19, cex=0.5 )
	lines(x.test, f.test, col=4,lwd=2)
	lines(  x.test, colMeans (fit_draw$f_predict))
	for( s in 1:1000)
		lines(x.test,  fit_draw$f_predict[s,], col='grey', lwd=0.2)
	fit_draw_predict=extract(fit, pars="f_predict", permuted=F)
	par(mfrow=c(2,4), mar=c(1,1,1,0))
	for(k in 1:8){
		plot(x.test, y.test, col=   col.vec_test, pch=19, cex=0.5 )
		lines(x.test, f.test, col=4,lwd=2)
		for( s in 1:1000)
			lines(x.test,  fit_draw_predict[s,k,], col='grey', lwd=0.1)
		lines(x.test, f.test, col=4,lwd=2)
	}
	fit_draw_lik=extract(fit, pars="log_lik", permuted=F)
	fit_draw_lik_test=extract(fit, pars="log_lik_test", permuted=F)
	fit_draw_lp=extract(fit, pars="lp__", permuted=F)
	fit_draw_lp=fit_draw_lp[,,1]
	library(loo)
	options(mc.cores = 8)
	chains=8
	loo_elpd=matrix(NA,n, chains)
	for(i in 1:chains){
		log_lik_matrix=fit_draw_lik[,i,]
		loo_elpd[,i]=(loo(log_lik_matrix)$pointwise[,1])
	}
	#st_weight=stacking_weights( loo_elpd, lambda=1.0000)
	st_weight=stack_with_na( loo_elpd, lambda=1.00000001)$full_weight
	st_weight[is.na(st_weight)]=0
	stacking_loo_result=log_score_loo(w=st_weight, lpd_point= loo_elpd)
	loo_chain=apply(loo_elpd,  2, sum )*n.test/n
	log_lik_test_agg=test_aggregrate(fit_draw_lik_test)
	stacking_loo_test=log_score_loo(w=st_weight, lpd_point= log_lik_test_agg)
	unif_test=log_score_loo(w=rep(1/8, 8), lpd_point= log_lik_test_agg)
	ind_test = apply(log_lik_test_agg, 2, sum )
	bma_test=log_score_loo(BMA_weight(fit_draw_lp), lpd_point= log_lik_test_agg)
	pesudo_bma=log_score_loo(pesudo_BMA_weight_chain(loo_elpd), lpd_point= log_lik_test_agg)
	select_test=ind_test [which.max(loo_chain)]
	loo_test_max[sim_i,]=  c(stacking_loo_test, unif_test,bma_test, pesudo_bma,select_test, max_r )
}
print(loo_test_max)
## We expect the first column (stacking) to be the largest.




#  Graph: gp_stacking.pdf ---------------------------------
# save(sigma2_vec, concentrate_vec,loo_test_max_mean_concentrate,loo_test_max,  file= "~/Desktop/gp/t_reg_hpc_resultsnew.RData")
# load(file= "~/Desktop/gp/t_reg_hpc_resultsnew.RData")
pdf("~/Desktop/gp_stacking.pdf", width = 7, height=2.7)
layout(matrix(c(1:4),nrow=2), width = c(1,1),height = c(2.5, 1))
par(mar=c(0.5,1,1,2),oma=c(1.4,6,0.5,0),
		mgp=c(1.5,0.1,0), lwd=0.5,tck=-0.01, cex.axis=0.7, cex.lab=0.9, cex.main=0.9,pty='m', bty='l', xpd=F)   
win_over_unif=loo_test_max[,2]-loo_test_max[,1]
win_over_bma=loo_test_max[,3]-loo_test_max[,1]
win_over_pbma=loo_test_max[,4]-loo_test_max[,1]
win_over_select=loo_test_max[,5]-loo_test_max[,1]
plot( sigma2_vec, win_over_unif, ylim= c(-36, 3) , type='n', axes = F , xlab="", ylab="",yaxs='i') 
lines(sigma2_vec, win_over_unif, col="darkorange", lwd=1)
points(sigma2_vec, win_over_unif, col="darkorange", pch=13,cex=0.7)
lines(sigma2_vec, win_over_bma , col="darkblue", lwd=1)
points(sigma2_vec, win_over_bma , col="darkblue", pch=18,cex=0.7)
lines(sigma2_vec, win_over_pbma, col="darkgreen", lwd=1)
points(sigma2_vec, win_over_pbma, col="darkgreen", pch=17, cex=0.7)
lines(sigma2_vec, win_over_select, col="grey30", lwd=1)
points(sigma2_vec, win_over_select, col="grey30",pch=20,cex=0.5)
abline(h=0, col="darkred", lwd=1)
mtext(2, text ="relative test log  \n predictive densities", las=2,  cex=0.7, line = 1)
mtext(3, line = 0.5, text="varying observational noise", cex=0.7)
axis(2, las=2, at=c(-30,-20,-10,0), lwd=0.5 )
axis(2, las=2, at=0, lwd=0.5, labels = "stacking =  ", hadj = 1, col.axis = 'darkred' , cex.axis=0.7)
axis(1, at=c(0.2,0.4,0.6,0.8, 1), lwd=0.5)
axis(2,   las=2, lwd=0.5, at=c(-32,-2), hadj=2,labels = c("worse fit", "better fit"), tick = F, xpd=T)
text(1, -6, labels = "LOO selection", col="grey30", xpd=T, cex=0.7)
text(0.87, -13, labels = "pseudo-BMA", col="darkgreen", xpd=T, cex=0.7)
text(0.9, -20, labels = "BMA / IS", col="darkblue", xpd=T, cex=0.7)
text(1, -30, labels = "uniform", col="darkorange", xpd=T, cex=0.7)
box(lwd=0.5,bty='l')
abline(h=c(-30,-20,-10,0), lwd=0.5, lty=2, col='grey')
plot( sigma2_vec, loo_test_max[,6], ylim=c(1,1.22), type='n', axes = F , xlab="", ylab="",yaxs='i') 
abline(h=c(1.05, 1.1, 1.15, 1.2), lwd=0.5, lty=2, col='grey')
lines( sigma2_vec, loo_test_max[,6],lwd=1) 
points( sigma2_vec, loo_test_max[,6],cex=0.5,pch=20) 
axis(1, at=c(0.2,0.4,0.6,0.8, 1) , lwd=0.5 )
axis(2, at=c(1,1.05,1.1,1.2), las=2, lwd=0.5 )
mtext(1, text = expression(sigma[2]), cex=0.7, line = 1)
mtext(2, text ="max R hat", las=2,  cex=0.7, line = 2)
box(lwd=0.5,bty='l')
win_over_unif=loo_test_max_mean_concentrate[,2]-loo_test_max_mean_concentrate[,1]
win_over_bma=loo_test_max_mean_concentrate[,3]-loo_test_max_mean_concentrate[,1]
win_over_pbma=loo_test_max_mean_concentrate[,4]-loo_test_max_mean_concentrate[,1]
win_over_select=loo_test_max_mean_concentrate[,5]-loo_test_max_mean_concentrate[,1]
plot( concentrate_vec, win_over_unif, ylim= c(-12, 1) , type='n', axes = F , xlab="", ylab="",yaxs='i') 
lines(concentrate_vec, win_over_unif, col="darkorange", lwd=1)
points(concentrate_vec, win_over_unif, col="darkorange", pch=13,cex=0.7)
lines(concentrate_vec, win_over_bma, col="darkblue", lwd=1)
points(concentrate_vec, win_over_bma , col="darkblue", pch=18,cex=0.7)
lines(concentrate_vec, win_over_pbma, col="darkgreen", lwd=1)
points(concentrate_vec, win_over_pbma, col="darkgreen", pch=17, cex=0.7)
lines(concentrate_vec, win_over_select, col="grey30", lwd=1)
points(concentrate_vec, win_over_select, col="grey30",pch=20,cex=0.5)
abline(h=0, col="darkred", lwd=1)
mtext(3, line = 0.5, text="varying concentration factor", cex=0.7)
axis(2, las=2, at=c(-0,-3,-6,-9), lwd=0.5 )
axis(2, las=2, at=0, lwd=0.5, labels = "stacking=", hadj = 1.1, col.axis = 'darkred' , cex.axis=0.7)
axis(1, at=c(2,4,6,8) , lwd=0.5 )
abline(h=c(-3,-6,-9,0), lwd=0.5, lty=2, col='grey')
box(lwd=0.5,bty='l')
text(7.5, -1, labels = "LOO selection", col="grey30", xpd=T, cex=0.7)
text(7.5, -4, labels = "pseudo-BMA", col="darkgreen", xpd=T, cex=0.7)
text(8.2, -5.7, labels = "BMA / IS", col="darkblue", xpd=T, cex=0.7)
text(8, -8.8, labels = "uniform", col="darkorange", xpd=T, cex=0.7)
plot( concentrate_vec, loo_test_max_mean_concentrate[,6], ylim=c(1,1.22), type='n', axes = F , xlab="", ylab="",yaxs='i') 
abline(h=c(1.05, 1.1, 1.15, 1.2), lwd=0.5, lty=2, col='grey')
lines( concentrate_vec, loo_test_max_mean_concentrate[,6],lwd=1) 
points( concentrate_vec, loo_test_max_mean_concentrate[,6],cex=0.5,pch=20) 
axis(1, at=c(2,4,6,8) , lwd=0.5 )
axis(2, at=c(1,1.05,1.1,1.2), las=2, lwd=0.5 )
mtext(1, text = "concentration factor", cex=0.7, line = 1)
box(lwd=0.5,bty='l')
dev.off()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## A careful look: Why these chains are predictively different, even when R hat is mild? 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Case One: marginal diffeences are accumulated---------------------------------------  

n=20
concentrate= 5
set.seed(20948)
x=sort(runif(n,-3,3))
f=0.3 + 0.4*x + 0.5 * sin(2.7*x) + 1.1/(1 + x^2 )
n=20
sd1=0.1
outlier_proption=0.3
id_out=sort(sample(1:n, size=round(outlier_proption*n ), prob =  exp( - (((1:n)-.4*n  )/n*concentrate )  ^2)  ))
noise1=rnorm(n,0,1)

sd2=0.6
y=f+sd1 * noise1
y[id_out]=f[id_out]+abs(   sd2* noise1[id_out]   ) * rep(c(1,-1), round(length(id_out)/2) )
n.test= 1000
x.test=sort(runif(n.test,-3,3))
f.test=0.3 + 0.4*x.test + 0.5 * sin(2.7*x.test) + 1.1/(1 + x.test^2 )
id_out_test=sort(sample(1:n.test, size=round(outlier_proption*n.test ), prob =  exp( - (((1:n.test)-.4*n.test  )/n.test*concentrate )^2)  ))
y.test=f.test+rnorm(n.test, 0,sd1)
y.test[id_out_test]=f.test[id_out_test]+abs(  rnorm(length(id_out_test), 0,sd2)   ) * rep(c(1,-1), round(length(id_out_test)/2) )
col.vec=rep(1, n)
col.vec[id_out]=2
plot(x, y, col= col.vec )
lines(x, f, col=4)
col_vec_test=rep(1, n.test)
col_vec_test[id_out_test]=2
plot(x.test, y.test, col=   col_vec_test )
lines(x.test, f.test, col=4)
fit <- stan(file = "~/Desktop/gp/t_reg.stan", 
						data=list(N=length(x), x=x, y=y, N_test=length(x.test), 
											x_test=x.test,y_test=y.test),
						init_r=10 , chains = 8, iter = 8000,seed=5838298,
						control=list(max_treedepth=5) )

# save(fit, file="~/Desktop/gp/fit_Rhat_1_new.RData")

# Graph: gp_pointwise.pdf ==============================

load( file="~/Desktop/gp/fit_Rhat_1_new.RData")
print(fit)
temp=summary(fit)
max_r=max(temp$summary[,"Rhat"])
fit_draw_lik=extract(fit, pars="log_lik", permuted=F)
fit_draw_lik_test=extract(fit, pars="log_lik_test", permuted=F)
fit_draw_lp=extract(fit, pars="lp__", permuted=F)
fit_draw_lp=fit_draw_lp[,,1]
library(loo)
options(mc.cores = 8)
chains=8
loo_elpd=matrix(NA,n, chains)
for(i in 1:chains){
	log_lik_matrix=fit_draw_lik[,i,]
	loo_elpd[,i]=(loo(log_lik_matrix)$pointwise[,1])
}
#st_weight=stacking_weights( loo_elpd, lambda=1.0000)
st_weight=stack_with_na( loo_elpd, lambda=1.00000001)$full_weight
st_weight[is.na(st_weight)]=0
stacking_loo_result=log_score_loo(w=st_weight, lpd_point= loo_elpd)
loo_chain=apply(loo_elpd,  2, sum )*n.test/n
log_lik_test_agg=test_aggregrate(fit_draw_lik_test)
stacking_loo_test=log_score_loo(w=st_weight, lpd_point= log_lik_test_agg)
unif_test=log_score_loo(w=rep(1/8, 8), lpd_point= log_lik_test_agg)
ind_test = apply(log_lik_test_agg, 2, sum )
bma_test=log_score_loo(BMA_weight(fit_draw_lp), lpd_point= log_lik_test_agg)
pesudo_bma=log_score_loo(pesudo_BMA_weight_chain(loo_elpd), lpd_point= log_lik_test_agg)
select_test=ind_test [which.max(loo_chain)]

print(c(stacking_loo_test, unif_test,bma_test, pesudo_bma,select_test, max_r ))
## check stacking does better

## Compute pointwise difference among all chains using test data lpd
log_mean_exp=function(v)
{
	max(v)+ log(mean(exp(v- max(v) )))
}
pointwise_test_lpd=function(lpd_point_test)
{
	test_elpd=matrix(NA, ncol=nrow(lpd_point_test[1,,]), nrow=ncol((lpd_point_test[1,,])))
	S=dim(lpd_point_test)[2]
	for(i in 1:S)
		test_elpd[,i]= apply(lpd_point_test[,i,],  2, log_mean_exp)
	return(test_elpd)
}
pointwise_test_lpd_matrix=pointwise_test_lpd(fit_draw_lik_test) # pointwise test score per chain

mean_vec=apply(pointwise_test_lpd_matrix-pointwise_test_lpd_matrix[,1], 2, mean)
se_vec=apply(pointwise_test_lpd_matrix-pointwise_test_lpd_matrix[,1], 2, sd)/ sqrt(n.test)
chain_rank=order(mean_vec) ## reorder chains
se_vec=se_vec[order(mean_vec)]
mean_vec=mean_vec[order(mean_vec)]

## Compute pointwise difference among all chains using loo
loo3=loo(fit_draw_lik[,1,])  # use chain one as reference
loo_k=list()
for(i in 1:chains){
	log_lik_matrix=fit_draw_lik[,i,]
	loo_k[[i]]=loo(log_lik_matrix)
}	
loo_compare(loo_k)
mean_loo=c()
se_loo=c()
for(i in 1:8){
	lc=loo_compare(loo_k[[i]],loo3)
	lc=lc[order(as.numeric( substr (rownames(lc), 6,7))),]
	mean_loo[i]=lc[1,1]-lc[2,1]
	se_loo[i]=max(lc[,2])
}	
mean_loo=mean_loo[chain_rank]
se_loo=se_loo[chain_rank ]
mean_loo=mean_loo/n  ## renormalize to each point
se_loo=se_loo/n

fit_new=extract(fit, pars="f_predict", permuted=F)
fit_sigma=extract(fit, pars=c("sigma", "alpha", "rho"), permuted=F)
draw_poly=function(x_sim, f_sim)
{
	 index=sort(sample(1:1000,300))
	 x_sim=x_sim[index]
	 f_sim=f_sim[,index]
	lines(x_sim,  apply(f_sim, 2, mean), col='darkred')
	polygon(c(x_sim , rev(x_sim)), c(apply(f_sim, 2, quantile, 0.975) ,rev(apply(f_sim, 2, quantile, 0.025))) , col=alpha("darkred", alpha = 0.2), border=NA)
	polygon(c(x_sim , rev(x_sim)), c(apply(f_sim, 2, quantile, 0.75) ,rev(apply(f_sim, 2, quantile, 0.25))) , col=alpha("darkred", alpha = 0.45), border=NA)
}
fit_draw_predict=extract(fit, pars="f_predict", permuted=F)
f.test=0.3 + 0.4*x.test + 0.5 * sin(2.7*x.test) + 1.1/(1 + x.test^2 )
r_hat_vec=summary(fit)$summary[,"Rhat"]


# pointwise t test, adjusted for n_eff
t_test=function(x, y){
	n1=ess_mean(x)
	n2=ess_mean(y)
	s_pooled=sqrt( (n1*var(x)+ n2* var(y)) / (n1+n2)  )
	return(mean(x-y )/s_pooled /sqrt( 1/n1 + 1/n2 ))
}
p_value_from_t=function(t){
	2*pnorm(-abs(t))
}
fit_draw_predict_agg=extract(fit, pars="f_predict")$f_predict
t_score_f=t_score_f_pd=t_score_f_lpd=ess_f=c()
for(i in 1:1000)
{
	t_score_f[i]=t_test  (fit_new[,1,i] ,fit_new[,3,i])
}
for(k  in 2:8)
	print(mean(  pointwise_test_lpd_matrix[,k]>pointwise_test_lpd_matrix[,1]))
for(i in 1:1000)
{
	t_score_f_pd[i] =t_test(exp (fit_draw_lik_test[,8,i])  ,  exp(fit_draw_lik_test[,1,i]))
	t_score_f_lpd[i] =t_test(  (fit_draw_lik_test[,8,i])  ,   (fit_draw_lik_test[,1,i]))
	ess_f[i] =ess_mean(fit_draw_predict_agg[,i]  )/32000
}
col.vec=rep(1, n)
col.vec[id_out]=2
col.vec=alpha( c('darkblue', 'darkred'), alpha=0.4)[col.vec]


### pointwise comparison of predictive distributions of f and likelihood between chain 1 and 8
pdf("~/desktop/gp_pointwise.pdf", width = 3.3, height=2.7)
par(mfcol=c(3,1), mar=c(0,5.6,0.5,0.2),oma=c(2,0,1.5,0),
		mgp=c(1.5,0.1,0), lwd=0.5,tck=-0.01, cex.axis=0.9, cex.lab=0.9, cex.main=0.9,pty='m', bty='l', xpd=F)  
plot(x.test, t_score_f, type='l',  axes=F,xlab="",ylab=" ",lwd=1, xlim=c(-3,3), ylim=  c(-3.5,3.5))
abline(v=seq(-3,3,by=1), lwd=0.5,lty=2, col='grey')
abline(h=c(-2,0,2), lwd=0.5,lty=2, col=2)
mtext(2, text ="pointwise\n   t-score  of\n predicted f", las=2,  cex=0.7, line = 1.6)
axis(1, at=c(-3,0,3) ,  labels = NA, lwd=0.5)
axis(2, at=c(-2,0,2) , lwd = 0.5, las=2)
box(bty='l',lwd=0.5)
mtext(3, text="pointwise comparison of predictive distributions\n of f and likelihood between chain 1 and 8", cex=0.7, line=0)
points(x, x*0-3.5, pch=15, col=col.vec , cex=0.9)
plot(x.test, t_score_f_pd, type='l', lwd=0.8, col=alpha('grey20', alpha=0.8), axes=F,xlab="",ylab=" ", xlim=c(-3,3), ylim=  c(-3.5,3.5))
abline(v=seq(-3,3,by=1), lwd=0.5,lty=2, col='grey')
abline(h=c(-2,0,2), lwd=0.5,lty=2, col=2)
mtext(2, text ="pointwise\n   t-score of\n  test data\n likelihood", las=2,  cex=0.7, line = 1.6)
axis(1, at=c(-3,0,3) ,  labels = NA, lwd=0.5)
axis(2, at=c(-2,0,2) , lwd = 0.5, las=2)
box(bty='l',lwd=0.5)
points(x, x*0-3.5, pch=15, col=col.vec, cex=0.9)
plot(x.test, ess_f, type='l', lwd=1, axes=F,xlab="",ylab=" ", xlim=c(-3,3), ylim=  c(0,1.02),yaxs='i')
abline(v=seq(-3,3,by=1), lwd=0.5,lty=2, col='grey')
abline(h=c(0,0.5,1), lwd=0.5,lty=2, col='grey')
mtext(2, text ="relative\n  ESS", las=2,  cex=0.7, line = 1.6, padj =-.2)
axis(1, at=c(-3,0,3) , lwd=0.5)
axis(2, at=c(0,0.5,1) ,labels=c("0","0.5","1"), lwd = 0.5, las=2)
box(bty='l',lwd=0.5)
mtext(1, text = "x", cex=0.7, line = 1)
points(x, x*0+0.02, pch=15, col=col.vec, cex=0.9)
points(c(-5, -5),  c(0.18, 0.08), col=unique(col.vec),pch=15, cex=1, xpd=T)
text(c(-4.1, -4.1),  c(0.18, 0.08), labels = c("observed data ", " observed outliers"), cex=0.7, xpd=T)
dev.off()


# Graph: gp_pointwise.pdf ==============================================
#save(x, y, x.test, y.test, f.test, f, p_c,r_hat_vec, mean_vec, se_vec,  mean_loo, se_loo,  fit_new, file="~/Desktop/gp/gp_graph2_new.RData")
load(file="~/Desktop/gp/gp_graph2.RData")

#graph fitted result at n=20#
pdf("~/desktop/gp_fitted_n20", width = 7.5, height=2.5)#,unit="in", res=800 )
layout(matrix(c(1, 2, 3,4,5,6,7,7),nrow=2), width = c(1.1,1,1.5,1),height = c(1, 1))
par(mar=c(1,1.3,1,1.5),oma=c(1.2,0,1.3,0),
		mgp=c(1.5,0.1,0), lwd=0.5,tck=-0.01, cex.axis=0.9, cex.lab=0.9, cex.main=0.9,pty='m', bty='l', xpd=F)  
hist(r_hat_vec[c(1:(3+ n), length(r_hat_vec))], breaks =seq(0.99,1.06, by=0.005), axes=F,xlab="",ylab=" " , main="")
axis(1,  at=c(1,1.025, 1.05),labels = NA, line=-0.2, lwd=0, lwd.ticks = 0.5)
mtext(3, text="max R hat among all \n  sampling parameters = 1.05", cex=0.6, line=0)
abline(v=1.05, col='darkred')
text(y=c(2,2,4), x=r_hat_vec[1:3], labels = c(expression(rho),expression(alpha), expression(sigma) ))
axis(2, at=c(0,5,10),lwd=0.5,las=2, line = -0.7)
hist(r_hat_vec[-c(1:(3+ n), length(r_hat_vec))], breaks =seq(0.99,1.06, by=0.005),  axes=F,xlab="",ylab=" " , main="")
mtext(3, text="max R hat among all \n transformed  parameters = 1.025", cex=0.6, line=0)
axis(1,  at=c(1,1.025, 1.05),labels = c("1", "1.025","1.05"), line=-0.2, lwd=0, lwd.ticks = 0.5)
abline(v=1.05, col='darkred')
mtext(1, text = "R hat", cex=0.7, line = 1)
axis(2, at=c(0,400,800),lwd=0.5,las=2, line = -0.7)

plot(1:8, mean_vec,  axes=F,xlab="",ylab=" " ,ylim=range(mean_vec+se_vec, mean_vec-se_vec), pch=19, cex=0.5)
abline(h=0,lwd=0.5,lty=2, col='grey')
for( i in 1:8){
	lines(c(i,i), c(mean_vec[i] + se_vec[i],mean_vec[i] - se_vec[i]   ),lwd=1)
}
mtext(3, text="paired test-elpd \n difference  +/- se", cex=0.7, line=0)
axis(1, at=c(1:8), lwd=0.5, labels = NA, tick = -0.02)
axis(2, at=c(0,.04,.08), lwd=0.5,    las=2)
box(bty='l',lwd=0.5)


plot(1:8, mean_loo,  axes=F,xlab="",ylab=" " ,ylim=range(mean_loo+se_loo, mean_vec-se_loo), pch=19, cex=0.5)
abline(h=0,lwd=0.5,lty=2, col='grey')
for( i in 1:8){
	lines(c(i,i), c(mean_loo[i] + se_loo[i],mean_loo[i] - se_loo[i]   ),lwd=1)
}
mtext(3, text="paired loo-elpd \n difference  +/- se", cex=0.7, line=-0.3)
axis(1, at=c(1:8), lwd=0.5)
axis(2, at=c(0,0.04,.08), lwd=0.5, las=2)
mtext(1, text = "chains", cex=0.7, line = 1)
box(bty='l',lwd=0.5)

plot(x, y, pch=19, cex=1, col=alpha('darkblue',alpha = 0.5), axes=F,xlab="",ylab=" ", xlim=c(-3,3), ylim=c(-2.5,3))  
abline(v=seq(-3,3,by=1), lwd=0.5,lty=2, col='grey')
mtext(2, text ="y", las=2,  cex=0.7, line = 1)
mtext(3, line = 0, text="posterior predictive distribution \n of f in chain 1", cex=0.7)
lines(x.test, f.test,lwd=1, col="darkblue")
draw_poly(x_sim=x.test, f_sim=fit_new[,1,])
axis(1, at=c(-3,0,3) , lwd=0.5)
axis(2, las=2, at=c(-2,0,2), lwd=0.5)
box(bty='l',lwd=0.5)

plot(x, y, pch=19, cex=1, col=alpha('darkblue',alpha = 0.5), axes=F,xlab="",ylab=" ", xlim=c(-3,3), ylim=c(-2.5,3)) 
abline(v=seq(-3,3,by=1), lwd=0.5,lty=2, col='grey')
draw_poly(x_sim=x.test, f_sim=fit_new[,8,])
mtext(1, text = "x", cex=0.7, line = 1)
mtext(2, text ="y", las=2,  cex=0.7, line = 1)
lines(x.test, f.test,lwd=1, col="darkblue")
axis(1, at=c(-3,0,3) , lwd=0.5)
axis(2, las=2, at=c(-2,0,2), lwd=0.5)
box(bty='l',lwd=0.5)
mtext(3, line = -0.5, text="chain 8", cex=0.7)

index=sample(1:4000,2000)
plot(fit_new[index ,3,750],  fit_new[index,3,550], col=alpha('darkred', alpha = 0.2), pch=20,cex=0.5, axes=F,xlab="",ylab=" ") 
mtext(1, text = " f | x=1.5", cex=0.7, line = 1)
mtext(2, text = "f | x=0.3", cex=0.7, line = 1)
mtext(3, text = "posterior draws of \n f in chain 1", cex=0.7, line = 0)
box(bty='l',lwd=0.5)
axis(2, at=c(1,1.5, 2) , las=2,lwd=0.5)
axis(1, at=c(0,1,2) , lwd=0.5)
dev.off()

#Case 2: bad chains from bad init_r==========================================
set.seed(20948)
n=40
sd1=0.2
sd2=1
outlier_proption=0.4
concentrate=5
x=sort(runif(n,-3,3))
f=0.3 + 0.4*x + 0.5 * sin(2.7*x) + 1.1/(1 + x^2 )
id_out=sample(1:n, size=round(outlier_proption*n ), prob =  exp( - (((1:n)-.4*n  )/n*concentrate )  ^2)  )
y=f+rnorm(n, 0,sd1)
y[id_out]=f[id_out]+abs(  rnorm(length(id_out), 0,sd2)   ) * rep(c(1,-1), round(length(id_out)/2) )

n.test= 300
x.test=sort(runif(n.test,-3,3))
f.test=0.3 + 0.4*x.test + 0.5 * sin(2.7*x.test) + 1.1/(1 + x.test^2 )
id_out_test=sample(1:n.test, size=round(outlier_proption*n.test ), prob =  exp( - (((1:n.test)-.4*n.test  )/n.test*concentrate )^2)  )
y.test=f.test+rnorm(n.test, 0,sd1)
y.test[id_out_test]=f.test[id_out_test]+abs(  rnorm(length(id_out_test), 0,sd2)   ) * rep(c(1,-1), round(length(id_out_test)/2) )
fit <- stan(file = "~/Desktop/gp/t_reg.stan", 
						data=list(N=length(x), x=x, y=y, N_test=length(x.test), 
											x_test=x.test,y_test=y.test),
						init_r=10 , chains = 8, iter = 8000,seed=5838298,
						control=list(max_treedepth=5) )

#save(fit, file="~/Desktop/gp/stiff.RData")


#Graph: gp_fitted_n40.pdf==========================================
load(file="~/Desktop/gp/stiff.RData")
fit_draw_predict=extract(fit, pars="f_predict", permuted=F)

fit_draw_lik=extract(fit, pars="log_lik", permuted=F)
fit_draw_lik_test=extract(fit, pars="log_lik_test", permuted=F)
fit_draw_lp=extract(fit, pars="lp__", permuted=F)
fit_draw_lp=fit_draw_lp[,,1]


pointwise_test_lpd_matrix=pointwise_test_lpd(fit_draw_lik_test) # pointwise test score per chain
apply(pointwise_test_lpd_matrix, 2,mean)
mean_vec=apply(pointwise_test_lpd_matrix-pointwise_test_lpd_matrix[,6], 2, mean)
se_vec=apply(pointwise_test_lpd_matrix-pointwise_test_lpd_matrix[,6], 2, sd)/ sqrt(n.test)
se_vec=se_vec[order(mean_vec)]
mean_vec=mean_vec[order(mean_vec)]
col.vec=rep(alpha('darkblue',alpha = 0.5), n)
col.vec[12]=alpha('darkred',alpha = 0.5)
pch_vec=rep(1,  n)
pch_vec[12]=19
fit_new_f=extract(fit, pars="f_predict", permuted=F)
pdf("~/desktop/gp_r_25.pdf", width = 7.5, height=2) 
layout(matrix(c(1:5),nrow=1), width = c(1, 1.2,1.2,1,1),height = c(1))
par(oma=c(1 ,0.6,1.6,0.3), pty='m',mar=c(1,0.5,1.5,0.5) ,mgp=c(1,0.25,0), lwd=1,tck=-0.02, cex.axis=0.9, cex.lab=0.9, cex.main=0.9,xpd=F, bty='l')
plot(1:8, mean_vec,  axes=F,xlab="",ylab=" " ,ylim=range(mean_vec+se_vec, mean_vec-se_vec), pch=19, cex=0.5)
abline(h=0,lwd=0.5,lty=2, col='grey')
for( i in 1:8){
	lines(c(i,i), c(mean_vec[i] + se_vec[i],mean_vec[i] - se_vec[i]   ),lwd=1)
}
mtext(3, text="paired mean-elpd \n difference  +/- se", cex=0.7, line=0)
axis(1, at=c(1:8), lwd=0.5)
axis(2, at=c(0,15,30), lwd=0.5, las=2)
mtext(1, text = "chains", cex=0.7, line = 1)
box(bty='l',lwd=0.5)
plot(x, y, pch=19, cex=1, col=alpha('darkblue',alpha = 0.5), axes=F,xlab="",ylab=" ", xlim=c(-3,3), ylim=c(-2,3))  
abline(v=seq(-3,3,by=1), lwd=0.5,lty=2, col='grey')
mtext(2, text ="y", las=2,  cex=0.7, line = 1)
mtext(3, line = 0, text="posterior predictive \n distribution  of f \n with a good initialization ", cex=0.7)
lines(x.test, f.test,lwd=1, col="darkblue")
draw_poly(x_sim=x, f_sim=fit_new[,2,])
axis(1, at=c(-3,0,3) , lwd=0.5)
axis(2, las=2, at=c(-2,0,2), lwd=0.5)
box(bty='l',lwd=0.5)
mtext(1, text = "x", cex=0.7, line = 1)
plot(x, y, pch=19, cex=0.4, col=alpha('darkblue',alpha = 0.5), axes=F,xlab="",ylab=" ", xlim=c(-3,3), ylim=c(-100,100)) 
abline(v=seq(-3,3,by=1), lwd=0.5,lty=2, col='grey')
draw_poly(x_sim=x, f_sim=fit_new[,6,])
mtext(1, text = "x", cex=0.7, line = 1)
mtext(2, text ="y", las=2,  cex=0.7, line = 1)
lines(x.test, f.test,lwd=1, col="darkblue")
axis(1, at=c(-3,0,3) , lwd=0.5)
axis(2, las=2, at=c(-100,0,100), lwd=0.5)
box(bty='l',lwd=0.5)
mtext(3, line = 0, text="with a bad initialization, \n posterior draws of f  \n are fluctuant", cex=0.7)
fit_new=extract(fit, pars="f", permuted=F)
# for( s in 1:1000){
# 	lines(x,  fit_new[s,6,], col=alpha('grey',alpha = 0.5), lwd=0.1)
#   points(x,  fit_new[s,6,], col=alpha('grey',alpha = 0.5), pch=20)
# }
hist(fit_new[,6,12], probability = TRUE,breaks =20,axes=F,xlim=c(0.291,0.293), xlab="",ylab=" " , main="")
abline(v=y[12], col='darkblue',lwd=1.5)
axis(1, labels ="observed y[12]", at=y[12], col.ticks = 'darkblue', col.axis = 'darkblue', cex.axis=0.7, hadj = 0.5, lwd=0, tick = -0.45, lwd.ticks =1.5)
axis(1, at=c(0.291,0.292,0.293), lwd=0,lwd.ticks = 0.5,line=-0.5)
mtext(3, text="posterior draws of f[12] \n in the bad chain \n overfit  y[12]", cex=0.7, line=0)
mtext(1, text = "posterior draws of f[12]", cex=0.7, line = 1)
sigma_6=log(extract(fit, pars=c("sigma"), permuted=F, inc_warmup =T)[,6,])
MC_mean_sigma=cumsum(sigma_6)/c(1:length(sigma_6))
par(mar=c(1,2.5,1.5,0.5))
graph_index=c(1:200,  sort(sample(201:7999,500)), 8000)
plot(c(1:8000)[graph_index],MC_mean_sigma[graph_index], lwd=1, type='l',axes=F,xlab="",ylab=" ",ylim=c(-9.914, -9.912)) 
mtext(3, text="posterior draws of sigma \n in the bad chain \n   are nearly 0", cex=0.7, line=0)
axis(2, at=c(-9.914, -9.912) , lwd= 0.5, las=2)
mtext(2, text = "runing  \n average \n of log sigma", cex=0.6, line = 0, las=2)
axis(1, at=c(0,4000,8000), lwd=0.5)
axis(1, at=c(2000,6000), lwd=0, labels =c("warm \n up",  "kept \n samples"), cex.axis=0.9,line=-1.5)
mtext(1, text ="iterations",   cex=0.7, line = 1)
box(bty='l',lwd=0.5)
dev.off()


