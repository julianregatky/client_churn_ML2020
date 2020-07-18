splines_matrix <- function(dataset) {
  features.num <- dataset[,unlist(lapply(dataset, is.numeric))]
  mat <- lapply(features.num, function(x) as.data.frame(bs(x,knots = quantile(x)[2:4], degree = 3))) %>% bind_cols()
  return(mat)
}

s <- suppressWarnings
