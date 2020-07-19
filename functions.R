splines_matrix <- function(dataset) {
  features.num <- dataset[,unlist(lapply(dataset, is.numeric))]
  mat <- lapply(features.num, function(x) as.data.frame(bs(x,knots = quantile(x)[2:4], degree = 3))) %>% bind_cols()
  return(mat)
}

f1.score <- function(pred,actual,cutoff) {
  conf.matrix <- table(pred = pred < cutoff,actual = actual < cutoff)
  precision <- conf.matrix[2,2]/sum(conf.matrix[2,])
  recall <- conf.matrix[2,2]/sum(conf.matrix[,2])
  f1_score <- 2*(precision*recall)/(precision+recall)
}

s <- suppressWarnings
