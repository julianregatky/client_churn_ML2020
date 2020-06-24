rm(list=ls())

# ~~~~~~~~~~~~~~ GENERAL SETTINGS ~~~~~~~~~~~~~~~~~

# Cargamos todas las librerías necesarias
# pacman las carga y, de no estar instaladas, previamente las instala
if (!require('pacman')) install.packages('pacman')
pacman::p_load(tidyverse,mlr,glmnet,pROC)

# Fijamos el working directory
setwd('/Users/julianregatky/Documents/GitHub/client_churn_ML2020')

# Importamos el dataset
dataset <- read.table('train.csv', header = T, sep =',', dec = '.')

# ~~~~~~~ ANÁLISIS EXPLORATORIO Y FEATURE ENGINEERING ~~~~~~~~~~

# Eliminamos features cuasi-constantes
dataset <- removeConstantFeatures(dataset,
                                  perc = 0.01, # Fijamos threshold del 99%
                                  dont.rm = 'TARGET')

# Unificamos features duplicados
del_col <- c()
for(i in 1:(ncol(dataset)-1)) {
  for(j in (i+1):ncol(dataset)) {
    if(identical(dataset[,i],dataset[,j])) {
      # identificamos columnas idénticas en el data.frame
      del_col <- c(del_col,j)
    }
  }
}
dataset <- dataset[,setdiff(1:ncol(dataset),del_col)] # Eliminamos una de las duplicadas

# La variable 'nac' es la única que cuenta con datos faltantes (56 obs con NA)
# Esto representa sólo ~0.17% de las obs. Imputamos datos faltantes con el valor 2
# que es el reportado para la variable en ~97.8% de los casos.
sum(is.na(dataset$nac))/length(dataset$nac) # ~0.17%
sum(dataset$nac[!is.na(dataset$nac)] == 2)/sum(!is.na(dataset$nac)) # ~97.8%

dataset$nac[is.na(dataset$nac)] <- 2

# ~~~~~~~~~~~~~~ MODELOS ~~~~~~~~~~~~~

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

# Agregamos términos polinómicos
features.polynomial <- polynomial(dataset %>% select(-TARGET), n = 3)

set.seed(123)
index_train <- sample(1:nrow(dataset),round(nrow(dataset)*0.9))

###########################
###       LASSO         ###
###########################

dataset.lasso <- cbind(dataset,features.polynomial)

x_train <- dataset.lasso[index_train,] %>% select(-TARGET)
x_validation <- dataset.lasso[setdiff(1:nrow(dataset.lasso),index_train),] %>% select(-TARGET)

y_train <- dataset.lasso[index_train,'TARGET']
y_validation <- dataset.lasso[setdiff(1:nrow(dataset.lasso),index_train),'TARGET']

# Cross-Validation, 10 folds
x_train_matrix <- model.matrix( ~ .-1, x_train)
cv.out = cv.glmnet(x_train_matrix, y_train, alpha = 1, nfolds = 10)
plot(cv.out)
lambda_star = cv.out$lambda.min

# Estimamos el modelo con el lambda de CV
model.lasso = glmnet(x = x_train_matrix,
                     y = y_train,
                     family = 'binomial',
                     alpha = 1, # LASSO
                     lambda = lambda_star)
coef(model.lasso, s = lambda_star)

x_validation_matrix <- model.matrix( ~ .-1, x_validation)
pred.lasso = predict(model.lasso, s = lambda_star , newx = x_validation_matrix)
roc_obj <- suppressWarnings(roc(y_validation,pred.lasso))
auc(roc_obj)
