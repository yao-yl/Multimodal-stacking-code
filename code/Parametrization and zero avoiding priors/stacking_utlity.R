##
##required functions for stacking

log_score_loo <- function(w, lpd_point) {
  sum <- 0
  N=dim(lpd_point)[1]
  for (i in 1: N  ) {
    sum <- sum + log(exp(lpd_point[i, ]) %*% w)
  }
  return(as.numeric(sum))
}





stacking_weights=function(lpd_point, lambda=1.0001)
{
  m=stan_model("optim.stan")
  S=dim(lpd_point)[2]
  s_w=optimizing(m,  data = list(N=dim(lpd_point)[1], K=S, lpd_point=lpd_point, lambda=rep(lambda, dim(lpd_point)[2])), iter=3000)$par[1:S] 
  return(s_w)
}  




stack_with_na=function(lpd_point, lambda=1.0001, lpd_point_test=NULL){
  S=dim(lpd_point)[2]
  N=dim(lpd_point)[1]
  flag=rep(0,S)
  for( i in 1:S)
    if(  sum( is.na( lpd_point[, i])) ==0)
      flag[i]=1
  lpd_point=lpd_point[,which(flag==1)]
  st_weight=stacking_weights( lpd_point=lpd_point, lambda=lambda)
  full_weight=rep(NA,S)
  full_weight[which(flag==1)]=st_weight
  loo_score=log_score_loo(st_weight, lpd_point)
  if (!is.null(lpd_point_test) ){
    lpd_point_test=lpd_point_test[,which(flag==1)]
    test_score=log_score_loo(st_weight, lpd_point_test)
  }
  else
    test_score=NULL
  return(list(loo_score=loo_score, full_weight=full_weight, test_score=test_score, flag=flag))
}


remove_dso_filename=function(stan_model_name){
  dso_filename = stan_model_name@dso@dso_filename
  loaded_dlls = getLoadedDLLs()
  if (dso_filename %in% names(loaded_dlls)) {
    message("Unloading DLL for model dso ", dso_filename)
    model.dll = loaded_dlls[[dso_filename]][['path']]
    dyn.unload(model.dll)
  } else {
    message("No loaded DLL for model dso ", dso_filename)
  }
  
  
  loaded_dlls = getLoadedDLLs()
  loaded_dlls = loaded_dlls[str_detect(names(loaded_dlls), '^file')]
  if (length(loaded_dlls) > 10) {
    for (dll in head(loaded_dlls, -10)) {
      message("Unloading DLL ", dll[['name']], ": ", dll[['path']])
      dyn.unload(dll[['path']])
    }
  }
  message("DLL Count = ", length(getLoadedDLLs()), ": [", str_c(names(loaded_dlls), collapse = ","), "]")
}


stacked_effective_sample_size=function(n_eff_per_chain,full_weight)
{
  full_weight= full_weight/sum( full_weight)
  return(   1/sum(  1/n_eff_per_chain *  full_weight^2)   )
}
