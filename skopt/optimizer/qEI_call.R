qEI_approx = function(mean, covar_mat, current_max_obj){
  cormat <- cov2cor(covar_mat) # 
  cc <- qEI_cpp(mu = mean, s = sqrt(diag(covar_mat)), cormat, current_max_obj)
  return(cc)
}