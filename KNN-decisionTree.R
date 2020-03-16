###########################
# ALDA: hw2.R 
# Instructor: Dr. Thomas Price
# Mention your team details here
# @author: Yang Shi/yangatrue and Krishna Gadiraju/kgadira
# Nirav Shah nshah25
# Neel Parik nnparik2
# Sai Santosh Balusu sbalusu
############################

require(caret)
require(rpart)

# ------- Part A -------

calculate_euclidean <- function(p, q) {
  # Input: p, q are numeric vectors of the same length
  # output: a single value of type double, containing the euclidean distance between p and q.
  return(sum((p-q)**2)**0.5)
  
}

calculate_cosine <- function(p, q) {
  # Input: p, q are numeric vectors of the same length
  # output: a single value of type double, containing the cosine distance between p and q.
  return (sum(p*q)/(sum(p**2)**0.5 * sum(q**2)**0.5))
  
}

knn <- function(x_train, y_train, x_test, distance_method = 'euclidean', k = 3){
  # You will be IMPLEMENTING a KNN Classifier here
  
  # You can re-use the calculate_euclidean and calculate_cosine methods from HW1 here.
  # for each row in the distance matrix, calculate the 'k' nearest neighbors
  # and return the most frequently occurring class from these 'k' nearest neighbors.
  
  # INPUT:
  # x_train: Matrix with dimensions: (number_training_samples x number_features)
  # y_train: Vector with length number_training_samples of type factor - refers to the class labels
  # x_test: Matrix with dimensions: (number_test_samples x number_features)
  # k: integer, represents the 'k' to consider in the knn classifier
  # distance_method: String, can be of type ('euclidean' or 'cosine')
  
  # OUTPUT:
  # A vector of predictions of length = number of samples in y_test and of type factor.
  
  # NOTE 1: For cosine, remember, you are calculating similarity, not distance. As a result, K nearest neighbors 
  # k values with highest values from the distance_matrix, not lowest. 
  # For euclidean, you are calculating distance, so you need to consider the k lowest values. 
  
  # NOTE 2:
  # In case of conflicts, choose the class with lower numerical value
  # E.g.: in 4NN, if you have 2 NN of class 1, 2 NN of class 2, there is a conflict b/w class 1 and class 2
  # In this case, you will choose class 1. 
  
  # NOTE 3:
  # You are not allowed to use predefined knn-based packages/functions. Using them will result in automatic zero.
  # Allowed packages: R base, utils
  
  y_test <- vector(length = dim(x_test)[1])
  if(distance_method == 'euclidean')
    d_mat <- t(apply(x_test,1,function(test_r) t(apply(x_train,1,function(train_r) calculate_euclidean(train_r,test_r)))))
    
  else if (distance_method == "cosine")
    d_mat <- t(apply(x_test,1,function(test_r) t(apply(x_train,1,function(train_r) calculate_cosine(train_r,test_r)))))
  
  for(i in 1:dim(x_test)[1])
  {
    distance <- data.frame("dist" = d_col[i,], "class"= y_train)
    dis <- distance[order(-distance$dist), ]
    k_neighbors <- dis[1:k,"class"]
    keys <- unique(k_neighbors)
    max_count <- tabulate(match(k_neighbors, keys))
    y_test[i] <- min(c(keys[max_count == max(max_count)]))
  
  }
  
  return(factor(y_test, ordered = FALSE, labels = levels(y_train)))
  
}

# ------- Part B -------

dtree <- function(x_train, y_train, x_test){
  # You will build a CART decision tree, then use the model to predict class values for a test dataset.
  
  # INPUT:
  # x_train: Matrix with dimensions: (number_training_samples x number_features)
  # y_train: Vector with length number_training_samples of type factor - refers to the class labels
  # x_test: Matrix with dimensions: (number_test_samples x number_features)
  # n_folds: integer, refers to the number of folds for n-fold cross validation
  
  # OUTPUT:
  # A vector of predictions of length = number of samples in y_test and of type factor.
  
  # Allowed packages: rpart, R Base, utils
  
  # HINT1: Make sure to read the documentation for the rpart package. Check out the 'rpart' and 'predict' functions.
  
  # HINT2: I've given you attributes and class labels as separate variables. Do you need to combine them 
  # into a data frame for rpart?
  
  df <- data.frame(x_train, y_train)
  dec_tree <- rpart(y_train~., data = df, parms = list(split="information"))
  return(predict(dec_tree, newdata = x_test, type = "class"))
  
}

# ------- Part C -------

generate_k_folds <- function(n, k) {
  # This function should randomly assign n datapoints to k folds
  # for cross validation. We will use this function in 
  # k_fold_cross_validation_prediction below.
  
  # INPUT:
  # n: Total number of samples.
  # k: The number of cross validation folds (e.g. 10 for 10-fold CV)
  
  # OUTPUT:
  # A vector representing the "fold" assignment of each row in
  # a dataset with n rows (e.g. 1 = first fold, 2 = second fold)
  # Example output:
  # > generate_k_folds(10, 3)
  # [1] 3 3 1 1 3 2 2 2 1 1
  
  # Note: Your folds should be *random*. You can use the sample() function among others
  # to achieve random assignment
  
  # Tip 1: You may wish to look at the function rep() if you don't want to use loops here.
  # Tip 2: R supports logical indexing. For example, If you input x = 1:5; y=c(1,2,3,2,1); 
  # x[y==2] gives an output of c(2,4).
  return(sample(x=rep(1:k, times = ceiling(n/k)), size=n, replace=FALSE))
  
}

k_fold_cross_validation_prediction <- function(x, y, k, k_folds, classifier_function) {
  # You will be IMPLEMENTING a cross validation predictor here.
  
  # INPUT:
  # x: Dataframe with dimensions: (number_samples x number_features)
  # y: Vector with length number_samples of type factor - refers to the class labels
  # k: The total fold number of cross validation.
  # k_folds: a vector representing the "fold" assignment of each row in the dataset, generated with
  # the generate_k_folds function
  # classifier_function: The classifier function you wish to use, it can be either knn or dtree. 
  # Note that you don't need to use quote marks for the functions as parameters.
  # OUTPUT:
  # A vector of predicted class values for each instance in x (length = nrow(x)). The ith
  # prediction should correspond to the ith row in x.
  y_pred <- vector(length = nrow(x))
  for(i in 1:k)
  {
    ind_train <- vector()
    ind_test <- vector()
    for(j in 1:nrow(x))
    {
      if (k_folds[j] != i) 
        ind_train <- append(ind_train, j)
      else 
        ind_test <- append(ind_test, j)
    }
    y_pred[ind_test] <- classifier_function(x[ind_train, ], y[ind_train], x[ind_test, ])
  }
  y_pred <- factor(y_pred, ordered = FALSE, labels = levels(y))
  return(y_pred)
  
}

# ------- Part D -------

calculate_confusion_matrix <- function(y_pred, y_true){
  # Given the following:
  
  # INPUT:
  # y_pred: predicted class labels (vector, each value of type factor)
  # y_true: ground truth class labels (vector, each value of type factor)
  
  # OUTPUT:
  # a confusion matrix of class "table" with Prediction to the left, and Reference on the top:
  # TN FN 
  # FP TP
  # You can use the caret library to calculate this confusion matrix.
  
  return(confusionMatrix(data = y_pred, reference = y_true)$table)
  
}

calculate_accuracy <- function(confusion_matrix){
  # Given the following:
  
  # INPUT:
  # confusion_matrix: A confusion matrix
  
  # OUTPUT:
  # prediction accuracy
  
  return((confusion_matrix[1]+confusion_matrix[4])/(confusion_matrix[1]+confusion_matrix[2]+confusion_matrix[3]+confusion_matrix[4]))

}

calculate_recall <- function(confusion_matrix){
  # Given the following:
  
  # INPUT:
  # confusion_matrix: A confusion matrix
  
  # OUTPUT:
  # prediction recall
  
  return(confusion_matrix[1]/(confusion_matrix[1]+confusion_matrix[2]))

}

calculate_precision <- function(confusion_matrix){
  # Given the following:
  
  # INPUT:
  # confusion_matrix: A confusion matrix
  
  # OUTPUT:
  # prediction precision
  
  return(confusion_matrix[1]/(confusion_matrix[1]+confusion_matrix[3]))

}

