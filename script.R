rm(list=ls())

# ~~~~~~~~~~~~~~ GENERAL SETTINGS ~~~~~~~~~~~~~~~~~

# Cargamos todas las librerías necesarias
# pacman las carga y, de no estar instaladas, previamente las instala
if (!require('pacman')) install.packages('pacman')
pacman::p_load(tidyverse,mlr,glmnet,ROCR,splines,rpart,randomForest,gbm,tictoc)

# Fijamos el working directory
#setwd('/Users/julianregatky/Documents/GitHub/client_churn_ML2020')
setwd('/home/julian/Documents/GitHub/client_churn_ML2020')

# Importamos el dataset
dataset <- read.table('train.csv', header = T, sep =',', dec = '.')

# Cargamos funciones propias
source('functions.R')

tic()
# ~~~~~~~ ANÁLISIS EXPLORATORIO Y FEATURE ENGINEERING ~~~~~~~~~~

# Eliminamos ID
dataset <- dataset %>% select(-ID)

# Eliminamos features cuasi-constantes
dataset <- removeConstantFeatures(dataset,
                                  perc = 0.01, # Fijamos threshold del 99%
                                  dont.rm = 'TARGET')

# Unificamos features duplicados
del_col <- c(); dataset_full <- dataset %>% na.omit()
for(i in 1:(ncol(dataset_full)-1)) {
  for(j in (i+1):ncol(dataset_full)) {
    if(cor(dataset_full[,i],dataset_full[,j]) > 0.99) {
      # identificamos columnas idénticas en el data.frame
      del_col <- c(del_col,j)
    }
  }
}
dataset <- dataset[,setdiff(1:ncol(dataset),del_col)] # Eliminamos una de las duplicadas

ct <- 0
for(i in 1:ncol(dataset)) {
  most_freq <- prop.table(table(dataset[,i])) %>% .[order(desc(.))] %>% .[1]
  if(most_freq > 0.9 & colnames(dataset)[i] != 'TARGET') {
    dataset[,i] <- as.numeric(as.character(dataset[,i]) == names(most_freq))
    ct <- ct + 1
    cat('\rVariables binarizadas:',ct)
  }
}

# La variable 'nac' es la única que cuenta con datos faltantes (56 obs con NA)
# Esto representa sólo ~0.17% de las obs. Imputamos datos faltantes con el valor TRUE
# que es el reportado para la variable en ~97.8% de los casos.
sum(is.na(dataset$nac))/length(dataset$nac) # ~0.17%
sum(dataset$nac[!is.na(dataset$nac)] == TRUE)/sum(!is.na(dataset$nac)) # ~97.8%

dataset$nac[is.na(dataset$nac)] <- 1

# ~~~~~~~~~~~~~~ MODELOS ~~~~~~~~~~~~~

set.seed(123) # Por replicabilidad
index_train <- sample(1:nrow(dataset),round(nrow(dataset)*0.8)) # Muestra de training

###########################
###       LASSO         ###
###########################

# Control local polinómico de hasta grado 3 (splines cúbicos)
features.spline <- splines_matrix(dataset[,setdiff(colnames(dataset),'TARGET')])

dataset.lasso <- cbind(dataset,features.spline)

x_train <- dataset.lasso[index_train,] %>% select(-TARGET)
x_test <- dataset.lasso[setdiff(1:nrow(dataset.lasso),index_train),] %>% select(-TARGET)

y_train <- dataset.lasso[index_train,'TARGET']
y_test <- dataset.lasso[setdiff(1:nrow(dataset.lasso),index_train),'TARGET']

# Cross-Validation, 10 folds
x_train_matrix <- model.matrix( ~ .-1, x_train)
cv.out = cv.glmnet(x_train_matrix, y_train, alpha = 1, nfolds = 5,
                   family = 'binomial', type.measure = 'auc')
plot(cv.out)
lambda_star = cv.out$lambda.1se

# Estimamos el modelo con el lambda de CV
model.lasso = glmnet(x = x_train_matrix,
                     y = y_train,
                     family = 'binomial',
                     alpha = 1, # LASSO
                     lambda = lambda_star,
                     standardize = TRUE)

x_test_matrix <- model.matrix( ~ .-1, x_test)
pred.lasso = predict(model.lasso, s = lambda_star , newx = x_test_matrix, type = 'response')

# AUC
performance(prediction(pred.lasso,y_test),"auc")@y.values[[1]]
auc_lasso <- performance(prediction(pred.lasso,y_test),"tpr","fpr")

###########################
###    Random Forest    ###
###########################
rm(list = setdiff(ls(),c('dataset','index_train','s','model.lasso','auc_lasso','pred.lasso')))

# Separamos en training, validation y testing sets (testing set idem antes)
test <- dataset[setdiff(1:nrow(dataset),index_train),]
train <- dataset[index_train,]

set.seed(123)
full_grid <- expand.grid(mtry = 10:50, ntree = seq(1000,3000,100), maxnodes = seq(20,100,5))
random_grid <- full_grid[sample(1:nrow(full_grid),20),]
best_auc <- 0
for(i in 1:nrow(random_grid)) {
  random.forest <- s(randomForest(TARGET ~.,
                                data = train,
                                mtry = random_grid$mtry[i],
                                ntree = random_grid$ntree[i],
                                maxnodes = random_grid$maxnodes[i],
                                importance = T,
                                proximity = F))
  pred.rforest = random.forest$predicted
  cat(i,'|',paste(colnames(random_grid),random_grid[i,],collapse = ' - '),'| auc oob:',performance(prediction(pred.rforest,train$TARGET),"auc")@y.values[[1]],'\n')
  if(performance(prediction(pred.rforest,train$TARGET),"auc")@y.values[[1]] > best_auc) {
    best_model <- random.forest
    best_auc <- performance(prediction(pred.rforest,train$TARGET),"auc")@y.values[[1]]
  }
}

model.rforest <- best_model
pred.rforest = predict(best_model,newdata=test)

# AUC
performance(prediction(pred.rforest,factor(test$TARGET)),"auc")@y.values[[1]] #AUC
auc_rforest <- performance(prediction(pred.rforest,test$TARGET),"tpr","fpr")

###########################
###        GBM          ###
###########################
rm(list = setdiff(ls(),c('dataset','index_train','s','model.lasso','auc_lasso','pred.lasso','model.rforest','auc_rforest','pred.rforest')))

# Separamos en training, validation y testing sets (testing set idem antes)
set.seed(123)
test <- dataset[setdiff(1:nrow(dataset),index_train),]
index_validation <- sample(index_train,nrow(test)/2) # Separamos obs del training set para validación
train <- dataset[setdiff(index_train, index_validation),]
validation <- dataset[index_validation,]

full_grid <- expand.grid(n.trees = seq(500,5000,100), shrinkage = seq(0.001,0.01,0.001), interaction.depth = 2:10, train.fraction = seq(0.5,0.9,0.1), bag.fraction = seq(0.5,0.9,0.1))
random_grid <- full_grid[sample(1:nrow(full_grid),20),]
best_auc <- 0
for(i in 1:nrow(random_grid)) {
  model.gbm = gbm(TARGET ~ .,
                data = train, 
                distribution = 'bernoulli',
                n.trees = random_grid$n.trees[i],
                shrinkage = random_grid$shrinkage[i],
                interaction.depth = random_grid$interaction.depth[i],
                train.fraction = random_grid$train.fraction[i],
                bag.fraction = random_grid$bag.fraction[i],
                verbose = F)
  pred.gbm = predict(model.gbm,newdata=validation)
  cat(i,'|',paste(colnames(random_grid),random_grid[i,],collapse = ' - '),'| auc-validation:',performance(prediction(pred.gbm,validation$TARGET),"auc")@y.values[[1]],'\n')
  if(performance(prediction(pred.gbm,validation$TARGET),"auc")@y.values[[1]] > best_auc) {
    best_model <- model.gbm
    best_auc <- performance(prediction(pred.gbm,validation$TARGET),"auc")@y.values[[1]]
  }
}

best_gbm <- best_model

pred.gbm = predict(best_model,newdata=test, type="response")
performance(prediction(pred.gbm,test$TARGET),"auc")@y.values[[1]] #AUC
auc_gbm <- performance(prediction(pred.gbm,test$TARGET),"tpr","fpr")

gbm.perf(best_gbm)

# ~~~~~~~~~~~~~~ COMPARATIVA ~~~~~~~~~~~~~
toc()

plot(auc_lasso, main = 'Curva de ROC')
points(auc_rforest@x.values[[1]],auc_rforest@y.values[[1]], type = 'l', col = 'red')
points(auc_gbm@x.values[[1]],auc_gbm@y.values[[1]], type = 'l', col = 'blue')
legend(0.56,0.25, legend = c('LASSO','Random Forest','GBM'), col = c('black','red','blue'),
       lty=rep(1,3), cex=0.6)

acc.lasso <- sum(diag(prop.table(table(test$TARGET == 0,pred.lasso <= sum(train$TARGET)/length(train$TARGET)))))
acc.rf <- sum(diag(prop.table(table(test$TARGET == 0,pred.rforest <= sum(train$TARGET)/length(train$TARGET)))))
acc.gbm <- sum(diag(prop.table(table(test$TARGET == 0,pred.gbm <= sum(train$TARGET)/length(train$TARGET)))))

ggplot(data = data.frame(est = c(as.vector(pred.lasso), as.vector(pred.rforest), as.vector(pred.gbm)),
                         actual = rep(factor(test$TARGET),3),
                         Model = c(rep('1. LASSO',nrow(test)),rep('2. Random Forest',nrow(test)),rep('3. GBM',nrow(test)))) %>%
         #filter(log(est) > -0.75) %>%
         mutate(Churn = ifelse(actual == 0,'No Churn','Churn')),
       aes(x = est, fill = Churn, alpha = 0.8)) +
  geom_density() +
  facet_wrap(~Model) +
  labs(title = 'Distribución de las clases según probabilidad predicha',
       fill = '',
       x = 'log(Probabilidad predicha)',
       y = 'Densidad') +
  theme_bw() +
  guides(alpha = FALSE)
ggsave(filename = 'distrib.png')
