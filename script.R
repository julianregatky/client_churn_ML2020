rm(list=ls())

# ~~~~~~~~~~~~~~ GENERAL SETTINGS ~~~~~~~~~~~~~~~~~

# Cargamos todas las librerías necesarias
# pacman las carga y, de no estar instaladas, previamente las instala
if (!require('pacman')) install.packages('pacman')
pacman::p_load(tidyverse,mlr,glmnet)

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




