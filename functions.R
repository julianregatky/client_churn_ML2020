# Toma un dataset, selecciona las variables numéricas y devuelve un
# data.frame con esas columnas elevadas a la i, tal que i=2,...,n
polynomial <- function(dataset, n = 3)
{
  features.num <- dataset[,unlist(lapply(dataset, is.numeric))]
  ret = dataset[,1]
  for(i in 2:n) {
    var.cols <- features.num^i; colnames(var.cols) <- paste0(colnames(var.cols),'_',i)
    ret <- cbind(ret,var.cols)
  }
  return(ret[,-1])
}

splines_matrix <- function(dataset) {
  features.num <- dataset[,unlist(lapply(dataset, is.numeric))]
  mat <- lapply(features.num, function(x) as.data.frame(bs(x,knots = quantile(x)[1:3], degree = 3))) %>% bind_cols()
  return(mat)
}

s <- suppressWarnings
