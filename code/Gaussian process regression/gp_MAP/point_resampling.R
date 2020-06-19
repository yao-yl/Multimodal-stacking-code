#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## stacking in GP for mode-based approximation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
source("stacking_utlitity.R")

## data from Neal
data.neal=read.table("~/Desktop/gp/odata.txt")
# odata.txt: Data from Neal's Software 
# http://llrforge.in2p3.fr/svn/cms/cms/trunk/Software/fbm.2004-11-10/doc/manual
colnames(data.neal)=c("x","y")

## The first half: training
x=data.neal$x [1:100]
y=data.neal$y [1:100]

# We use the second half as  validation to avoid LOO in mode-searching
x_val=data.neal$x [101:200]
y_val=data.neal$y [101:200]

# independent test data 
set.seed(100)
n_test=200
x_test=sort(rnorm(n_test, 0, 1))
f_test=0.3 + 0.4*x_test + 0.5 * sin(2.7*x_test) + 1.1/(1 + x_test^2 )
id.out=sort(sample(1:n_test, 0.05*n_test))
y_test=f_test+rnorm(n_test, 0,0.1)
y_test[id.out]=f_test[id.out]+rnorm(length(id.out), 0,1)
plot(x_test, y_test)
x_test=c(x_val, x_test) 
# For code simplicty combine these two vector,
# but the actual test data x_test[201:400] shall not be revealed until the method evaluation 
y_test=c(y_val, y_test)
n_test=300

n=length(x)
f=0.3 + 0.4*x + 0.5 * sin(2.7*x) + 1.1/(1 + x^2 )

m_three_pars=stan_model("~/Desktop/gp/opt_gp_three_pars.stan")
fit_three_pars <- sampling(m_three_pars, data=list(N=length(x), x=x, y=y), chains = 1, iter=10, seed=5838298)
#  you should find two modes:
opt_1= optimizing(m_three_pars, 
									data=list(N=n, x=x, y=y),
									init=list(log_rho= (1), 
														log_alpha=0.7, 
														log_sigma=0.1),
									hessian=T,
									seed=2020 ) 
opt_1_par=opt_1$par
print(opt_1_par)
##
##rho     alpha     sigma 
##0.9884577 1.8481367 0.2576907 

opt_2= optimizing(m_three_pars, 
									data=list(N=n, x=x, y=y),
									init=list(log_rho= -1, 
														log_alpha=-5, 
														log_sigma=log(5)),
									hessian=T,
									seed=2020)
opt_2_par=opt_2$par
print(opt_2_par)
##
# rho     alpha     sigma 
# 0.4766149 1.1859285 0.2358455 

### Laplace approx ==================================

grid_normal=function( n=1000){  # generate 3-D N(0,I)
	temp=matrix(NA,  3, n)
	for(i in 1:3)
		temp[i,]=rnorm(n,0,1)
	return(temp)
}
set.seed(2020)
row_grid_nomral=grid_normal()
find_grid=function(row_grid=row_grid_nomral,  opt){
	x1=opt$par[1:3]
	S=dim(row_grid_nomral)[2]
	shift=rbind( rep(x1[1],S), rep(x1[2],S) , rep(x1[3],S) )
	h1=opt$hessian
	cov1=solve(-h1)	
	eig_d= eigen(cov1)
	grid= eig_d$vectors %*% diag( sqrt(eig_d$values))    %*%    row_grid
	grid=t(grid)+t(shift)
	rownames(grid)=c()
	colnames(grid)=names(x1)
	return(grid)
}
grid_Laplace_1=find_grid(row_grid=row_grid_nomral,  opt=opt_1)
grid_Laplace_2=find_grid(row_grid=row_grid_nomral,  opt=opt_2)
S=dim(grid_Laplace_1)[1]
grid_lp1=grid_lp2=c()
for(i in 1: S)
{
	grid_lp1[i]=log_prob(fit_three_pars, upars=grid_Laplace_1[i,]) 
	grid_lp2[i]=log_prob(fit_three_pars, upars=grid_Laplace_2[i,]) 
}
exp(max(grid_lp1)-max(grid_lp2))
m_loo=stan_model("~/Desktop/gp/loo.stan")
laplace_gqs1=gqs(m_loo, data=list(N=length(x), x=x, y=y, 
																	N_test=length(x_test), 
																	x_test=x_test, y_test=y_test),
								 draws= expand_draws(grid_Laplace_1) )

laplace_gqs2=gqs(m_loo, data=list(N=length(x), x=x, y=y, 
																	N_test=length(x_test), 
																	x_test=x_test, y_test=y_test),
								 draws= expand_draws(grid_Laplace_2) )


fit_draw_lik1=extract(laplace_gqs1, pars="log_lik")$log_lik
fit_draw_lik2=extract(laplace_gqs2, pars="log_lik")$log_lik
fit_draw_test1=extract(laplace_gqs1, pars="log_lik_test")$log_lik_test
fit_draw_test2=extract(laplace_gqs2, pars="log_lik_test")$log_lik_test

fit_draw_lik=array(NA, c(dim(fit_draw_lik1)[1],  2,    dim(fit_draw_lik1)[2]))
fit_draw_lik[,1,]=fit_draw_lik1
fit_draw_lik[,2,]=fit_draw_lik2



fit_draw_lik_test=array( NA, c(dim(fit_draw_test1)[1],  2,  dim(fit_draw_test1)[2]))
fit_draw_lik_test[,1,]=fit_draw_test1
fit_draw_lik_test[,2,]=fit_draw_test2

# find stacking weights use the validation set
log_lik_test_agg=test_aggregrate(fit_draw_lik_test)
validata_agg=log_lik_test_agg[1:100,]   
st_weight=stacking_weights(validata_agg)


w_mode=c(exp(max(grid_lp1)-max(grid_lp2)), 1)
w_mode=w_mode/sum(w_mode)

w_is=c(exp(mean(grid_lp1)-mean(grid_lp2)), 1)
w_is=w_is/sum(w_is)

log_score_test_laplace=c( log_score_loo(w=st_weight, lpd_point= log_lik_test_agg[101:300,]),
													log_score_loo(w=w_mode, lpd_point= log_lik_test_agg[101:300,]),
													log_score_loo(w=w_is, lpd_point= log_lik_test_agg[101:300,]))

# grid_approx and inportance resampling==================================
grid_uniform=function(lower=-3.2, upper=3.2, n_0=30 ){
	temp=matrix(NA,  3, n_0^3)
	xx=seq(lower,upper,length.out = n_0)
	for (i in c(1:n_0))
		for (j in c(1:n_0))
			for (l in c(1:n_0))
			   temp[,(i-1)*n_0^2+(j-1)*n_0+l]= c(xx[i], xx[j], xx[l])
	return(temp)
}
row_grid=grid_uniform()

find_grid=function(row_grid=row_grid,  opt){
	x1=opt$par[1:3]
	S=dim(row_grid)[2]
	shift=rbind( rep(x1[1],S), rep(x1[2],S) , rep(x1[3],S) )
	h1=opt$hessian
	cov1=solve(-h1)	
	eig_d= eigen(cov1)
	grid= eig_d$vectors %*% diag( sqrt(eig_d$values))    %*%    row_grid
	grid=t(grid)+t(shift)
	rownames(grid)=c()
	colnames(grid)=names(x1)
	return(grid)
}


grid_unif_1=find_grid(row_grid=row_grid,  opt=opt_1)
grid_unif_2=find_grid(row_grid=row_grid,  opt=opt_2)
S=dim(grid_unif_1)[1]
grid_lp1_unif=grid_lp2_unif=c()
for(i in 1: S)
{
	grid_lp1_unif[i]=log_prob(fit_three_pars, upars=grid_unif_1[i,]) 
	grid_lp2_unif[i]=log_prob(fit_three_pars, upars=grid_unif_2[i,]) 
}

threshold=log(max(grid_lp1_unif, grid_lp2_unif))-log(10) #at least 0.1 * mode height
id.remain1= sample(1:S, 1000,  prob =  exp(grid_lp1_unif-max(grid_lp1_unif))* c(grid_lp1_unif> threshold)  )
id.remain2= sample(1:S, 1000,  prob =  exp(grid_lp2_unif-max(grid_lp2_unif))* c(grid_lp2_unif> threshold)   )

grid_unif_resample1= grid_unif_1[id.remain1,]
grid_unif_resample2=grid_unif_2[id.remain2,]

expand_draws=function(grid)
{
	expand_draw_matrix=grid
	temp= dimnames (as.matrix(fit_three_pars))
	temp[[2]]=temp[[2]][1:3]
	dimnames(expand_draw_matrix) = temp
	return(expand_draw_matrix)
}
unif_gqs1=gqs(m_loo, data=list(N=length(x), x=x, y=y, 
																	N_test=length(x_test), 
																	x_test=x_test, y_test=y_test),
								 draws= expand_draws(grid=grid_unif_resample1) )

unif_gqs2=gqs(m_loo, data=list(N=length(x), x=x, y=y, 
																	N_test=length(x_test), 
																	x_test=x_test, y_test=y_test),
								 draws= expand_draws(grid=grid_unif_resample2) )
fit_draw_lik1=extract(unif_gqs1, pars="log_lik")$log_lik
fit_draw_lik2=extract(unif_gqs2, pars="log_lik")$log_lik
fit_draw_test1=extract(unif_gqs1, pars="log_lik_test")$log_lik_test
fit_draw_test2=extract(unif_gqs2, pars="log_lik_test")$log_lik_test
fit_draw_lik=array(NA, c(dim(fit_draw_lik1)[1],  2,    dim(fit_draw_lik1)[2]))
fit_draw_lik[,1,]=fit_draw_lik1
fit_draw_lik[,2,]=fit_draw_lik2
fit_draw_lik_test=array( NA, c(dim(fit_draw_test1)[1],  2,  dim(fit_draw_test1)[2]))
fit_draw_lik_test[,1,]=fit_draw_test1
fit_draw_lik_test[,2,]=fit_draw_test2

log_lik_test_agg=test_aggregrate(fit_draw_lik_test)
validata_agg=log_lik_test_agg[1:100,]
st_weight=stacking_weights(validata_agg)
w_mode=c(exp(max(grid_lp1)-max(grid_lp2)), 1)
w_mode=w_mode/sum(w_mode)
w_is=c(exp(mean(grid_lp1)-mean(grid_lp2)), 1)
w_is=w_is/sum(w_is)
log_score_test_unif=c( log_score_loo(w=st_weight, lpd_point= log_lik_test_agg[101:300,]),
											 log_score_loo(w=w_mode, lpd_point= log_lik_test_agg[101:300,]),
											 log_score_loo(w=w_is, lpd_point= log_lik_test_agg[101:300,]))
### MAP-II =======================================================
test_stan=stan_model('~/Desktop/gp/test_gauss.stan')
mode1_data <- list(alpha= opt_1_par[5], rho= opt_1_par[4], sigma= opt_1_par[6], N=length(x), x=x, y=y,
									 N_test=length(x_test), x_test=x_test, y_test=y_test)
fit1 <- sampling(test_stan, data=mode1_data, iter=3000, warmup=0,
								 chains=1, seed=5838298, refresh=1000, algorithm="Fixed_param")
fit_draw_lik_test1=extract(fit1, pars="log_lik_test")$log_lik_test
fit_draw_lik1=extract(fit1, pars="log_lik")$log_lik
fit_draw_f=extract(fit1, pars="f_predict")$f_predict


mode2_data <- list(alpha= opt_2_par[5], rho= opt_2_par[4], sigma= opt_2_par[6], N=length(x), x=x, y=y,N_test=length(x_test), x_test=x_test, y_test=y_test)
fit2 <- sampling(test_stan, data=mode2_data, iter=3000, warmup=0,
									chains=1, seed=5838298, refresh=1000, algorithm="Fixed_param")
fit_draw_lik_test2=extract(fit2, pars="log_lik_test")$log_lik_test
fit_draw_lik2=extract(fit2, pars="log_lik")$log_lik
fit_draw_f2=extract(fit2, pars="f_predict")$f_predict
fit_draw_lik_test=array( NA, c(dim(fit_draw_lik_test1)[1],  2,  dim(fit_draw_lik_test1)[2]))
fit_draw_lik_test[,1,]=fit_draw_lik_test1
fit_draw_lik_test[,2,]=fit_draw_lik_test2
fit_draw_lik=array( NA, c(dim(fit_draw_lik1)[1],  2,  dim(fit_draw_lik1)[2]))
fit_draw_lik[,1,]=fit_draw_lik1
fit_draw_lik[,2,]=fit_draw_lik2
log_lik_test_agg=test_aggregrate(fit_draw_lik_test)
validata_agg=log_lik_test_agg[1:100,]
st_weight=stacking_weights(validata_agg, lambda=1.01)
w_mode= exp (c(opt_1$value, opt_2$value))
w_mode=w_mode/sum(w_mode)
log_score_test_mode=c( log_score_loo(w=st_weight, lpd_point= log_lik_test_agg[101:300,]),
													log_score_loo(w=w_mode, lpd_point= log_lik_test_agg[101:300,]),
													log_score_loo(w=w_is, lpd_point= log_lik_test_agg[101:300,]))
mlpd= rbind(log_score_test_mode,  log_score_test_laplace, log_score_test_unif)/200
colnames(mlpd)=c("stacking", "mode", "IS" )
print(mlpd) #: the last panel of gp_point.pdf



## graph ======================================================================================
# save(x, y, x_test, y_test, log_score_test_mode,  log_score_test_laplace, log_score_test_unif, file="~/Desktop/gp/mode_new.RData")
# save(x, y, x_test, y_test, log_score_test_mode,  log_score_test_laplace, log_score_test_unif,
# 		 log_alpha.grid, log_rho.grid,  exp_z, extract_pred, extract_pred2,
# 		 file="~/Desktop/gp/mode.RData")
# pdf("~/Desktop/gp_point.pdf",width=6.7,height=2.7)
# layout(matrix(c(1,1,2,3,4,4),nrow=2), width = c(1,1.5,1.5),height = c(1,1))
# 
# par( oma=c(0,1.3,1,0), pty='m',mar=c(2.5,1,1,1) , lwd=0.5,tck=-0.01, mgp=c(1.5,0.25,0), cex.axis=0.8, cex.lab=0.8, cex.main=0.9,xpd=F)
# 
# 
# 
# 
# contour(log_alpha.grid, log_rho.grid,  t(exp_z),levels=seq(0.15, 1, 0.1) * max(exp_z), drawlabels=FALSE, main="",axes=FALSE,xlab="",ylab=" ", xlim=c(-1, 3) , ylim=c(-1.0, 0.5))
# mtext(3, cex=0.7, text = "posterior density at \n sigma =0.25",line=-0.5)
# axis(1,  lwd=0.5, at=c(-1,1,3))
# axis(2,  lwd=0.5, at=c(-1,0,0.5,-0.5), las=2)
# box(lwd=0.5, bty='l')
# mtext(1, cex=0.7, text = expression(log~alpha), line=1.2)
# mtext(2, cex=0.7, text = expression(log~rho), line=.5, las=2)
# points(x1[2],x1[1],col="darkred", pch=3)
# points(x2[2],x2[1],col="darkblue", pch=4)
# text(x1[2]+1.5,x1[1]-0.1,col="darkred", labels = "mode 1")
# text(x2[2]+1.5,x2[1]-0.1,col="darkblue", labels = "mode 2")
# 
# 
# plot(x,y,col=alpha(1, alpha=0.6 ), pch=20 , cex=0.8,
# 		 main="",axes=FALSE,xlab="",ylab=" ", ylim=c(-1.5,2.5), xlim=c(-3,3))
# lines(x[order(x)], f[order(x)],col='gray30')
# f_9=apply(extract_pred$f_predict,  2, quantile, 0.975)
# f_1=apply(extract_pred$f_predict,  2, quantile, 0.025)
# f_7=apply(extract_pred$f_predict,  2, quantile, 0.75)
# f_2=apply(extract_pred$f_predict,  2, quantile, 0.25)
# f_5=apply(extract_pred$f_predict,  2, mean)
# lines( x_pred,  f_5, col=2 )
# polygon(c(x_pred ,rev(x_pred) ), c(f_1 ,rev(f_9)) , col=alpha("darkred", alpha = 0.3), border=NA)
# polygon(c(x_pred ,rev(x_pred) ), c(f_2 ,rev(f_7)) , col=alpha("darkred", alpha = 0.5), border=NA)
# mtext(3, cex=0.7, text = "prediction distribution \n of f at model 1", line=-0.5)
# axis(1,  lwd=0.5, at=c(-3,0,3))
# axis(2,  lwd=0.5, at=c(-1,0,1,2), las=2)
# box(lwd=0.5, bty='l')
# mtext(1, cex=0.7, text = "x", line=1)
# mtext(2, cex=0.7, text = "y", line=1.5, las=2)
# text(x=-0.2,y=-1, labels = "observed data", cex=.7, col='grey30')
# text(x=-2,y=-0.5, labels = "true f", cex=.7, col='grey30')
# 
# 
# plot(x,y,col=alpha(1, alpha=0.6 ), pch=20 , cex=0.8,
# 		 main="",axes=FALSE,xlab="",ylab=" ", ylim=c(-1.5,2.5), xlim=c(-3,3))
# lines(x[order(x)], f[order(x)],col='gray30')
# f_9=apply(extract_pred2$f_predict,  2, quantile, 0.975)
# f_1=apply(extract_pred2$f_predict,  2, quantile, 0.025)
# f_7=apply(extract_pred2$f_predict,  2, quantile, 0.75)
# f_2=apply(extract_pred2$f_predict,  2, quantile, 0.25)
# f_5=apply(extract_pred2$f_predict,  2, mean)
# lines( x_pred,  f_5, col=2 )
# polygon(c(x_pred ,rev(x_pred) ), c(f_1 ,rev(f_9)) , col=alpha("darkgreen", alpha = 0.3), border=NA)
# polygon(c(x_pred ,rev(x_pred) ), c(f_2 ,rev(f_7)) , col=alpha("darkgreen", alpha = 0.5), border=NA)
# mtext(3, cex=0.7, text = "f at model 2")
# axis(1,  lwd=0.5, at=c(-3,0,3))
# axis(2,  lwd=0.5, at=c(-1,0,1,2), las=2)
# box(lwd=0.5, bty='l')
# mtext(1, cex=0.7, text = "x", line=1)
# mtext(2, cex=0.7, text = "y", line=1.5, las=2)
# legend("bottomright", legend = c("50% CI", "95% CI"),  fill=alpha(c("darkgreen"), alpha=c( 0.8,0.3)), cex=0.8,border = NA,box.lty=0)
# 
#  
# 
# 
# 
# col_vec=c("darkred", "orange","darkblue")
# pch_vec=c(19,18,15)
# par( mar=c(3,3,1,4) )
# 
# plot(c(1:3),mlpd[,1], type='n', main="",axes=FALSE,xlab="",ylab=" ", ylim=range(mlpd))
# for(i in 3:1){
# points(c(1:3),mlpd[,i], col=col_vec[i], pch=pch_vec[i], cex=0.9)
# lines(c(1:3),mlpd[,i], col=col_vec[i], pch=20 ,cex=0.5)
# }
# axis(1,  lwd=0.5, at=c(1:3), labels = c("type-II \n MAP", "Laplace \n approx.", "Importance \n resampling" ),cex.axis=1, padj = 0.8 )
# axis(2,  lwd=0.5, at=c(-0.32, -0.28, -0.24), las=2)
# box(lwd=0.5, bty='l')
# mtext(2, cex=0.7, text = "mean \n test \n lpd", line=1.5, las=2)
# mtext(3, cex=0.7, text = "predictive performace  after\n  averaging two modes")
# text(x=rep(3.4,3),y=mlpd[3,], col=col_vec,  cex=1, labels = c("stacking", "mode \n height",   "importance\n weighting"), xpd=T)
# 
# 
# dev.off()
# 
