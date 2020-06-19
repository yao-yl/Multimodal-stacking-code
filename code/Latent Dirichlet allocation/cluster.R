.libPaths("/rigel/stats/users/yy2619/rpackages")
#args <- commandArgs(trailingOnly = TRUE)
args <-  Sys.getenv("SLURM_ARRAY_TASK_ID")
print(Sys.getenv("SLURM_ARRAY_JOB_ID"))
print(args)
arrayid <- as.integer(args[1])
set.seed(as.integer(arrayid))

library(rstan)
load(file="stan_input.RData")


library(rstan)
lda_fit=stan(file="lda.stan", data=data_input, iter=4000, thin = 20, control = list(max_treedepth = 15), seed=arrayid, chains = 1)
save(lda_fit,file=paste("arg_", arrayid, ".RData", sep=""))
rm(list=ls())