setwd("~/Desktop/mixture")
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


#####################
## normal mixture####
###(i)###############
set.seed(100)
mu1=5
mu2=-5
sigma=1
p=2/3
N=30
y=rep(NA, N)
y[1: round(N*p)]=rnorm(round(N*p), mu1,sigma)
y[-(1: round(N*p))]=rnorm(N-round(N*p), mu2,sigma)
n_chain=8
sm=stan_model(file="mixture.stan")
stan_fit_full=stan(file="mixture.stan", data = list (N=N,  y=y, sigma=sigma, p=p),  iter=3000, seed = 100,chains=n_chain) 
plot(extract(stan_fit_full)$mu1, extract(stan_fit_full)$mu2, col=alpha('darkred', alpha = 0.2), pch=19, cex=0.5  )
mean(extract(stan_fit_full)$mu1<=0)
fit_extract=extract(stan_fit_full, permuted=FALSE)
colMeans(fit_extract[,,   1]) # Chain 4:7 are sampled form the "bad" mode 
log_lik_mat=fit_extract[,,  (-(N:1)+dim(fit_extract)[3])  ]
loo_elpd=matrix(NA,N, n_chain)
for(i in 1:n_chain)
  loo_elpd[,i]=loo (log_lik_mat[,i,]) $pointwise[,1]

st_weight=stacking_weights( lpd_point=loo_elpd, optim_control=list(maxit=10000, abstol=1e-6))  # Chain 4:7 are sampled form the "bad" mode 
print( sum (st_weight$wts [4:7]) ) # the left mode: 0.001

mu_1_grid=seq(-13,13,length.out = 300)
mu_2_grid=seq(-13,13,length.out = 300)
grid_exp=expand.grid(mu_1_grid,mu_2_grid )
names(grid_exp)=c("mu1","mu2")

simulate_density=rep(0, dim(grid_exp)[1] )
for(i in 1:dim(grid_exp)[1] )
simulate_density[i]=log_prob(stan_fit_full, upars=c(grid_exp$mu1[i],grid_exp$mu2[i]))
plot(grid_exp$mu1,grid_exp$mu2,cex=0.05,col=alpha("darkblue",max(simulate_density)/simulate_density))


simulate_density_mu_1=rep(0, length(mu_1_grid) )

for(i in 1:length(mu_1_grid))
  simulate_density_mu_1[i]=  logSumExp(simulate_density[ which(grid_exp$mu1== mu_1_grid[i]   )])
simulate_density_normalize=simulate_density_mu_1-max(simulate_density_mu_1)+log(1.853) ## unnormalzied


right_mode_weight=exp(logSumExp(simulate_density_normalize[which(mu_1_grid>0)  ])-logSumExp(simulate_density_normalize))


pdf("mode2.pdf",width=8,height=1.5)
par(mfrow=c(1,5), oma=c(0 ,6.5,1.8,0 ), pty='m',mar=c(1,0.7,1,0.5) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.8, cex.main=0.9,xpd=F)
plot(0,  type='n', axes="F" , xlim=c(-10,10), ylim=c(0,1), yaxs='i')
abline(v=5, lwd=2.5)
axis(1,   at=c(-10,-5,0,5, 10),  labels =c("",-5,0,5, "") , lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
mtext(3, text = "true generating \n mechanism of mu1", cex=0.7,line=0.5)


plot(mu_1_grid, simulate_density_normalize, xlim=c(-10,10), type='l', ylim=c(-205,2) ,yaxs='i', axes="F", lwd=1, col="forestgreen")
axis(2,   las=2,  lwd=0.5, at=c(0,-100,-200,-300))
axis(1,   at=c(-10,-5,0,5, 10),  labels =c("",-5,0,5, "") , lwd=0.5, padj=-0.5)
abline(v=5, lwd=0.5,col=alpha("grey",alpha = 0.5), lty=2)
box(bty='l',lwd=0.5)
mtext(3, text = "exact log posterior density\n(unnormalized)", cex=0.7,line=0.5)
abline(h=max(simulate_density_normalize), lwd=0.5,col=alpha("grey",alpha = 0.5), lty=2)



plot(mu_1_grid, exp(simulate_density_normalize), xlim=c(-10,10), type='l', ylim=c(0,2) ,yaxs='i', axes="F", lwd=0.5, col="forestgreen")
polygon(c(mu_1_grid ,rev(mu_1_grid) ), c(exp(simulate_density_normalize) ,rep(0,length(mu_1_grid))) , col=alpha("forestgreen",alpha = 0.5), border=NA)
axis(1,   at=c(-10,-5,0,5, 10),  labels =c("",-5,0,5, "") , lwd=0.5, padj=-0.5)
abline(v=5, lwd=0.5,col=alpha("grey",alpha = 0.5), lty=2)
box(bty='l',lwd=0.5)
mtext(3, text = "exact posterior density \n(unnormalized)", cex=0.7,line=0.5)
text(0.5, 1, labels  = paste( "mass of\n right mode\n =", round(right_mode_weight, 3)) , cex=0.7, col="forestgreen")
 

nuts_density= density(adjust=0.2, extract(stan_fit_full)$mu1)
plot(nuts_density, col="darkblue" ,type='l' , lwd=0.5,axes=F,xlab="",ylab=" ",yaxs='i',  xlim=c(-10,10),ylim=c(0,2),main="")
x_trans= nuts_density$x
y_trans= nuts_density$y
polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=alpha("darkblue",alpha = 0.5), border=NA)
axis(2,   las=2,  lwd=0.5, at=c(0,-100,-200,-300))
axis(1,   at=c(-10,-5,0,5, 10),  labels =c("",-5,0,5, "") , lwd=0.5, padj=-0.5)
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))
box(bty='l',lwd=0.5)
mtext(3, text = "8 parallel chains \n unweighted", cex=0.7, line=0.5)
text(5, 1, labels  = "4 chains", cex=0.7, col="darkblue")
text(-5, 0.9, labels = "4 chains", cex=0.7,col="darkblue")
 

 
nuts_density= density(adjust=1.2, fit_extract[,1:3,1])
plot(nuts_density, col="darkred" ,type='l' , lwd=0.5,axes=F,xlab="",ylab=" ",yaxs='i' , xlim=c(-10,10),ylim=c(0,2),main="")
x_trans= nuts_density$x
y_trans= nuts_density$y
polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=alpha("darkred",alpha = 0.5), border=NA)
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))
axis(1,   at=c(-10,-5,0,5, 10),  labels =c("",-5,0,5, "") , lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
mtext(3, text = "stacking",  cex=0.7, line=0.5)
text(1, 1, labels  = paste( "mass of\n right mode\n =", round(sum(st_weight$wts[-(4:7)]), 3)) , cex=0.7, col="darkred")
mtext(2, text = "(ii)\n a bad mode\n normal mixtures",  adj = 0,  outer=T,cex=0.75,las=2, line=6.5)
 
dev.off()
################################################################




pdf("mode1.pdf",width=8,height=1.5)
par(mfrow=c(1,5), oma=c(0 ,6.5,1.8,0 ), pty='m',mar=c(1,0.7,1,0.5) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.8, cex.main=0.9,xpd=F)
plot(0,  type='n', axes="F" , xlim=c(-10,10), ylim=c(0,1), yaxs='i')
lines(c(-5, -5), y=c(0,1/3),  lwd=2.5)
lines(c(5, 5), y=c(0,2/3),  lwd=2.5)
axis(1,   at=c(-10,-5,0,5, 10),   labels =c("",-5,0,5, "") , lwd=0.5, padj=-0.5)

axis(2, tck=.01, at=c(0,1/3, 2/3),   labels=c("0",NA, NA),   lwd=0.5, las=2)
text(-9.4, c(1/3, 2/3),   c("1/3", "2/3"),  lwd=0.5, las=2, cex=.6)

box(bty='l',lwd=0.5)
mtext(3, text = "true generating \n mechanism of mu", cex=0.7,line=0.5)



simulate_density_2=dnorm( mean(y) , mu_1_grid, sd=sqrt(1/N), log = T)

plot(mu_1_grid, simulate_density_2, xlim=c(-10,10), type='l', ylim=c(-205,3) ,yaxs='i', axes="F", lwd=1, col="forestgreen")
axis(2,   las=2,  lwd=0.5, at=c(0,-100,-200,-300))
axis(1,   at=c(-10,-5,0,5, 10),  labels =c("",-5,0,5, "") , lwd=0.5, padj=-0.5)
abline(v=5, lwd=0.5,col=alpha("grey",alpha = 0.5), lty=2)
box(bty='l',lwd=0.5)
mtext(3, text = "exact log posterior density", cex=0.7,line=0.5)
 abline(v=c(-5, 5), lwd=0.5,lty=2, col="grey")

plot(mu_1_grid, exp(simulate_density_2), xlim=c(-10,10), type='l', ylim=c(0,2.3) ,yaxs='i', axes="F", lwd=0.5, col="forestgreen")
polygon(c(mu_1_grid ,rev(mu_1_grid) ), c(exp(simulate_density_2) ,rep(0,length(mu_1_grid))) , col=alpha("forestgreen",alpha = 0.5), border=NA)
axis(1,   at=c(-10,-5,0,5, 10),  labels =c("",-5,0,5, "") , lwd=0.5, padj=-0.5)
abline(v=5, lwd=0.5,col=alpha("grey",alpha = 0.5), lty=2)
box(bty='l',lwd=0.5)
mtext(3, text = "exact posterior density", cex=0.7,line=0.5)
abline(v=c(-5, 5), lwd=0.5,lty=2, col="grey")
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))


 
mtext(2, text = " (i)\n missing modes\n normal mixtures", adj = 0,  outer=T,cex=0.75,las=2, line=6.5)
 

dev.off()


 #######################################################



set.seed(100)
mu=c(-10,10)
n=100
y=rep(NA, n)
p=1/2
y[1:(n*p)]=rcauchy(n*(p),mu[1], 1)
y[(n*(p)+1):n]=rcauchy(n*(p),mu[2], 1)

plot(density(y))
mu_grid=true_density_log=seq(-17,17,length.out=2000)
for(i in 1:length(mu_grid) )
  true_density_log[i]=sum(dcauchy(y,mu_grid[i] ,1,log = T ))
true_density_log=true_density_log-max(true_density_log)
post=exp(true_density_log )
post_cdf <- cumsum(post)
plot(mu_grid,post)
plot(mu_grid,true_density_log)

K=8
fit_sample=stan("cauchy2.stan", data=list(n=n, y=y),chains = K , seed=100)

stan_samples=extract(fit_sample, permuted=F)
S=100
mu_sample=stan_samples[,,"mu"]


lpd_point=matrix(NA, S, K)
elpd_loo=c()
log_likelihood= stan_samples[,,2:(n+1)]
for(k in 1:K){
  L=loo( log_likelihood[,k,] )
  elpd_loo[k]=L$elpd_loo
  lpd_point[,k]  =L$pointwise[,1]    ## log(p_k (y_i | y_-i))
}
uwts <- exp( elpd_loo - max( elpd_loo))
w.loo1 <- uwts / sum(uwts)    


stackW=loo::stacking_weights(lpd_point,  optim_control=list(maxit=10000, abstol=1e-6))

round(cbind(stackW,  apply(mu_sample,2, mean)), digits = 3)



 
    
    xx= mu_grid
    yy=xx*0
    for( j in 1:8){
      density_est=  density(adjust=1, mu_sample[,j]) 
      # if( max(density_est$y)>100  )
      #   density_est=  density(beta_sim[,j,1], adjust=0.001)
      temp=approx(x=density_est$x, y=density_est$y, xout=xx)$y
      temp[is.na (temp)]=rep(0, sum(is.na (temp))) 
      yy=yy+stackW[j]*temp
    }
    stack_density_grid=yy


    yy=xx*0
    for( j in 1:8){
      density_est=  density(adjust=1, mu_sample[,j]) 
      # if( max(density_est$y)>100  )
      #   density_est=  density(beta_sim[,j,1], adjust=0.001)
      temp=approx(x=density_est$x, y=density_est$y, xout=xx)$y
      temp[is.na (temp)]=rep(0, sum(is.na (temp))) 
      yy=yy+1/8*temp
    }
    nuts_grid=yy


 
pdf("mode3.pdf",width=8,height=1.5)
par(mfrow=c(1,5), oma=c(0 ,6.5,1.8,0 ), pty='m',mar=c(1,0.7,1,0.5) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.8, cex.main=0.9,xpd=F)
plot(0,  type='n', axes="F" , xlim=c(-17,17), ylim=c(0,0.8), yaxs='i')
lines(c(-10, -10), y=c(0,1/2),  lwd=2.5)
lines(c(10, 10), y=c(0,1/2),  lwd=2.5)
axis(1,   at=c(-20, -10, 0,  10, 20),   labels =c("",-10,0,10, "") , lwd=0.5, padj=-0.5)

axis(2, tck=.01, at=c(0,0.5),   labels=c("0",NA),   lwd=0.5, las=2)
text(-16, 0.5,   "0.5",  lwd=0.5, las=2, cex=.6)

box(bty='l',lwd=0.5)
mtext(3, text = "true generating \n mechanism of mu", cex=0.7,line=0.5)





plot(mu_grid, true_density_log, xlim=c(-17,17), type='l', ylim=c(-200,1) ,yaxs='i', axes="F", lwd=1, col="forestgreen")
axis(2,   las=2,  lwd=0.5, at=c(0,-100,-200))
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)
mtext(3, text = "exact log posterior density\n(unnormalized)", cex=0.7,line=0.5)


plot(mu_grid, exp(true_density_log), xlim=c(-17,17), type='l', ylim=c(0,1.02) ,yaxs='i', axes="F", lwd=0.5, col="forestgreen")
polygon(c(mu_grid ,rev(mu_grid) ), c(exp(true_density_log) ,rep(0,length(true_density_log))) , col=alpha("forestgreen",alpha = 0.5), border=NA)
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)
mtext(3, text = "exact posterior density \n(unnormalized)", cex=0.7,line=0.5)

right_mode_weight=exp(logSumExp(  true_density_log[mu_grid>=0] )-logSumExp(  true_density_log ))

text(3, 0.7, labels  = paste( "mass of\n right mode\n =", round(right_mode_weight, 3)) , cex=0.7, col="forestgreen")
 





plot(mu_grid, nuts_grid, col="darkblue" ,type='l' , lwd=0.5,axes=F,xlab="",ylab=" ",yaxs='i',  xlim=c(-17,17),ylim=c(0,1.3),main="")
sum(apply(mu_sample,2, mean)<=0)
x_trans= mu_grid
y_trans= nuts_grid
polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=alpha("darkblue",alpha = 0.5), border=NA)
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))
box(bty='l',lwd=0.5)
mtext(3, text = "8 parallel chains \n unweighted", cex=0.7, line=0.5)
text(5, 1, labels  = "5 chains", cex=0.7, col="darkblue")
text(-8, 0.7, labels = "3 chains", cex=0.7,col="darkblue")
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)



plot(mu_grid,stack_density_grid,  col="darkred" ,type='l' , lwd=0.5,axes=F,xlab="",ylab=" ",yaxs='i' , xlim=c(-17,17),ylim=c(0,1.3),main="")
x_trans= mu_grid
y_trans= stack_density_grid
polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=alpha("darkred",alpha = 0.5), border=NA)
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
mtext(3, text = "stacking",  cex=0.7, line=0.5)
text(3, 1, labels  = paste( "mass of\n right mode\n =", round(sum( stackW [ apply(mu_sample,2, mean)>=0])
, 3)) , cex=0.7, col="darkred")
mtext(2, text = "(iii)\n a theoretically\n  good mode\n Cauchy mixtures \n w/equal weights",  adj = 0,  outer=T,cex=0.75,las=2, line=6.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)

dev.off()








###############################################
set.seed(100)
y[101]=rcauchy(1,mu[2], 1)
n=101
plot(density(y))
mu_grid=true_density_log=seq(-17,17,length.out=2000)
for(i in 1:length(mu_grid) )
  true_density_log[i]=sum(dcauchy(y,mu_grid[i] ,1,log = T ))
true_density_log=true_density_log-max(true_density_log)
post=exp(true_density_log )
post_cdf <- cumsum(post)
plot(mu_grid,post)
plot(mu_grid,true_density_log)

K=8
fit_sample=stan("cauchy2.stan", data=list(n=n, y=y),chains = K, seed=100)

stan_samples=extract(fit_sample, permuted=F)

mu_sample=stan_samples[,,"mu"]

lpd_point=matrix(NA, n, K)
elpd_loo=c()
log_likelihood= stan_samples[,,2:(n+1)]
for(k in 1:K){
  L=loo( log_likelihood[,k,] )
  elpd_loo[k]=L$elpd_loo
  lpd_point[,k]  =L$pointwise[,1]    ## log(p_k (y_i | y_-i))
}
uwts <- exp( elpd_loo - max( elpd_loo))
w.loo1 <- uwts / sum(uwts)    


stackW=loo::stacking_weights(lpd_point,  optim_control=list(maxit=10000, abstol=1e-6))
round(cbind(stackW,  apply(mu_sample,2, mean)), digits = 3)






xx= mu_grid
yy=xx*0
for( j in 1:8){
  density_est=  density(adjust=1, mu_sample[,j]) 
  # if( max(density_est$y)>100  )
  #   density_est=  density(beta_sim[,j,1], adjust=0.001)
  temp=approx(x=density_est$x, y=density_est$y, xout=xx)$y
  temp[is.na (temp)]=rep(0, sum(is.na (temp))) 
  yy=yy+stackW[j]*temp
}
stack_density_grid=yy


yy=xx*0
for( j in 1:8){
  density_est=  density(adjust=1, mu_sample[,j]) 
  # if( max(density_est$y)>100  )
  #   density_est=  density(beta_sim[,j,1], adjust=0.001)
  temp=approx(x=density_est$x, y=density_est$y, xout=xx)$y
  temp[is.na (temp)]=rep(0, sum(is.na (temp))) 
  yy=yy+1/8*temp
}
nuts_grid=yy



pdf("mode4.pdf",width=8,height=1.5)
par(mfrow=c(1,5), oma=c(0 ,6.5,1.8,0 ), pty='m',mar=c(1,0.7,1,0.5) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.8, cex.main=0.9,xpd=F)
plot(0,  type='n', axes="F" , xlim=c(-17,17), ylim=c(0,0.8), yaxs='i')
lines(c(-10, -10), y=c(0,1/2),  lwd=2.5)
lines(c(10, 10), y=c(0,1/2),  lwd=2.5)
axis(1,   at=c(-20, -10, 0,  10, 20),   labels =c("",-10,0,10, "") , lwd=0.5, padj=-0.5)
axis(2, tck=.01, at=c(0,0.5),   labels=c("0",NA),   lwd=0.5, las=2)
text(-16, 0.5,   "0.5",  lwd=0.5, las=2, cex=.6)
box(bty='l',lwd=0.5)
mtext(3, text = "true generating \n mechanism of mu", cex=0.7,line=0.5)

plot(mu_grid, true_density_log, xlim=c(-17,17), type='l', ylim=c(-200,1) ,yaxs='i', axes="F", lwd=1, col="forestgreen")
axis(2,   las=2,  lwd=0.5, at=c(0,-100,-200))
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)
mtext(3, text = "exact log posterior density\n(unnormalized)", cex=0.7,line=0.5)

plot(mu_grid, exp(true_density_log), xlim=c(-17,17), type='l', ylim=c(0,1.02) ,yaxs='i', axes="F", lwd=0.5, col="forestgreen")
polygon(c(mu_grid ,rev(mu_grid) ), c(exp(true_density_log) ,rep(0,length(true_density_log))) , col=alpha("forestgreen",alpha = 0.5), border=NA)
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)
mtext(3, text = "exact posterior density \n(unnormalized)", cex=0.7,line=0.5)
right_mode_weight=exp(logSumExp(  true_density_log[mu_grid>=0] )-logSumExp(  true_density_log ))
text(3, 0.7, labels  = paste( "mass of\n right mode\n =", round(right_mode_weight, 3)) , cex=0.7, col="forestgreen")
 
plot(mu_grid, nuts_grid, col="darkblue" ,type='l' , lwd=0.5,axes=F,xlab="",ylab=" ",yaxs='i',  xlim=c(-17,17),ylim=c(0,1.3),main="")
sum(apply(mu_sample,2, mean)<=0)
x_trans= mu_grid
y_trans= nuts_grid
polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=alpha("darkblue",alpha = 0.5), border=NA)
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))
box(bty='l',lwd=0.5)
mtext(3, text = "8 parallel chains \n unweighted", cex=0.7, line=0.5)
text(5, 1, labels  = "5 chains", cex=0.7, col="darkblue")
text(-8, 0.7, labels = "3 chains", cex=0.7,col="darkblue")
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)



plot(mu_grid,stack_density_grid,  col="darkred" ,type='l' , lwd=0.5,axes=F,xlab="",ylab=" ",yaxs='i' , xlim=c(-17,17),ylim=c(0,1.3),main="")
x_trans= mu_grid
y_trans= stack_density_grid
polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=alpha("darkred",alpha = 0.5), border=NA)
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
mtext(3, text = "stacking",  cex=0.7, line=0.5)
text(3, 1, labels  = paste( "mass of\n right mode\n =", round(sum( stackW [ apply(mu_sample,2, mean)>=0])
                                                              , 3)) , cex=0.7, col="darkred")
mtext(2, text = "(iv)\n an  ugly mode\n Cauchy mixtures\n w/equal weights\n and odd size",  adj = 0,  outer=T,cex=0.75,las=2, line=6.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)

dev.off()







###########################
#####v


set.seed(100)
mu=c(-10,10)
n=99
y=rep(NA, n)
p=1/3
y[1:(n*p)]=rcauchy(n*(p),mu[1], 1)
y[(n*(p)+1):n]=rcauchy(n*(p),mu[2], 1)

plot(density(y))
mu_grid=true_density_log=seq(-17,17,length.out=2000)
for(i in 1:length(mu_grid) )
  true_density_log[i]=sum(dcauchy(y,mu_grid[i] ,1,log = T ))
true_density_log=true_density_log-max(true_density_log)
post=exp(true_density_log )
post_cdf <- cumsum(post)
plot(mu_grid,post)
plot(mu_grid,true_density_log)

K=8

initf1 <- function(chain_id = 1) {list(mu=rnorm(1,0,50)) }
fit_sample=stan("cauchy2.stan", 
                data=list(n=n, y=y),chains = K , seed=100, init = initf1)

stan_samples=extract(fit_sample, permuted=F)
S=n
mu_sample=stan_samples[,,"mu"]
apply(mu_sample,2, mean)

lpd_point=matrix(NA, n, K)
elpd_loo=c()
log_likelihood= stan_samples[,,2:(n+1)]
for(k in 1:K){
  L=loo( log_likelihood[,k,] )
  elpd_loo[k]=L$elpd_loo
  lpd_point[,k]  =L$pointwise[,1]    ## log(p_k (y_i | y_-i))
}
uwts <- exp( elpd_loo - max( elpd_loo))
w.loo1 <- uwts / sum(uwts)    


stackW=loo::stacking_weights(lpd_point,  optim_control=list(maxit=10000, abstol=1e-6))

round(cbind(stackW,  apply(mu_sample,2, mean)), digits = 3)





xx= mu_grid
yy=xx*0
for( j in 1:8){
  density_est=  density(adjust=1, mu_sample[,j]) 
  # if( max(density_est$y)>100  )
  #   density_est=  density(beta_sim[,j,1], adjust=0.001)
  temp=approx(x=density_est$x, y=density_est$y, xout=xx)$y
  temp[is.na (temp)]=rep(0, sum(is.na (temp))) 
  yy=yy+stackW[j]*temp
}
stack_density_grid=yy


yy=xx*0
for( j in 1:8){
  density_est=  density(adjust=1, mu_sample[,j]) 
  # if( max(density_est$y)>100  )
  #   density_est=  density(beta_sim[,j,1], adjust=0.001)
  temp=approx(x=density_est$x, y=density_est$y, xout=xx)$y
  temp[is.na (temp)]=rep(0, sum(is.na (temp))) 
  yy=yy+1/8*temp
}
nuts_grid=yy



pdf("mode5.pdf",width=8,height=1.5)
par(mfrow=c(1,5), oma=c(0 ,6.5,1.8,0 ), pty='m',mar=c(1,0.7,1,0.5) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.8, cex.main=0.9,xpd=F)
plot(0,  type='n', axes="F" , xlim=c(-17,17), ylim=c(0,0.9), yaxs='i')
lines(c(-10, -10), y=c(0,1/3),  lwd=2.5)
lines(c(10, 10), y=c(0,2/3),  lwd=2.5)
axis(1,   at=c(-20, -10, 0,  10, 20),   labels =c("",-10,0,10, "") , lwd=0.5, padj=-0.5)

axis(2, tck=.01, at=c(0,1/3,2/3),   labels=c("0",NA,NA),   lwd=0.5, las=2)
text(-16, c(1/3,2/3),   c("1/3","2/3"),  lwd=0.5, las=2, cex=.6)

box(bty='l',lwd=0.5)
mtext(3, text = "true generating \n mechanism of mu", cex=0.7,line=0.5)





plot(mu_grid, true_density_log, xlim=c(-17,17), type='l', ylim=c(-300,1) ,yaxs='i', axes="F", lwd=1, col="forestgreen")
axis(2,   las=2,  lwd=0.5, at=c(0,-100,-200))
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)
mtext(3, text = "exact log posterior density\n(unnormalized)", cex=0.7,line=0.5)


plot(mu_grid, exp(true_density_log), xlim=c(-17,17), type='l', ylim=c(0,1.02) ,yaxs='i', axes="F", lwd=0.5, col="forestgreen")
polygon(c(mu_grid ,rev(mu_grid) ), c(exp(true_density_log) ,rep(0,length(true_density_log))) , col=alpha("forestgreen",alpha = 0.5), border=NA)
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)
mtext(3, text = "exact posterior density \n(unnormalized)", cex=0.7,line=0.5)

right_mode_weight=exp(logSumExp(  true_density_log[mu_grid>=0] )-logSumExp(  true_density_log ))

text(3, 0.7, labels  = paste( "mass of\n right mode\n =", round(right_mode_weight, 3)) , cex=0.7, col="forestgreen")
 





plot(mu_grid, nuts_grid, col="darkblue" ,type='l' , lwd=0.5,axes=F,xlab="",ylab=" ",yaxs='i',  xlim=c(-17,17),ylim=c(0,1.5),main="")
sum(apply(mu_sample,2, mean)<=0)
x_trans= mu_grid
y_trans= nuts_grid
polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=alpha("darkblue",alpha = 0.5), border=NA)
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))
box(bty='l',lwd=0.5)
mtext(3, text = "8 parallel chains \n unweighted", cex=0.7, line=0.5)
text(5, 1, labels  = "5 chains", cex=0.7, col="darkblue")
text(-8, 0.7, labels = "3 chains", cex=0.7,col="darkblue")
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)



plot(mu_grid,stack_density_grid,  col="darkred" ,type='l' , lwd=0.5,axes=F,xlab="",ylab=" ",yaxs='i' , xlim=c(-17,17),ylim=c(0,1.5),main="")
x_trans= mu_grid
y_trans= stack_density_grid
polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=alpha("darkred",alpha = 0.5), border=NA)
axis(2,   las=2,  lwd=0.5, at=c(0,1,2))
axis(1,   at=c(-10,0, 10),    lwd=0.5, padj=-0.5)
box(bty='l',lwd=0.5)
mtext(3, text = "stacking",  cex=0.7, line=0.5)
text(3, 1, labels  = paste( "mass of\n right mode\n =", round(sum( stackW [ apply(mu_sample,2, mean)>=0])
                                                              , 3)) , cex=0.7, col="darkred")
mtext(2, text = "(v)\n an ugly mode\n Cauchy mixtures \n unequal weights",  adj = 0,  outer=T,cex=0.75,las=2, line=6.5)
abline(v=c(10, -10),   lwd=0.5,col=alpha("grey", alpha = 0.5), lty=2)

dev.off()






 