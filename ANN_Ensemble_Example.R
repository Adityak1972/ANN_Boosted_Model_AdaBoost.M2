# Load the appropriate packages 
library(RSNNS)
detach("package:RSNNS", unload = T) # make sure that RSNNS is un attached initially because it conflicts with kohonen, later on we will use RSNNS when we are building the ANN base and boosted model 
library(mlbench)
library(caret)
library(caretEnsemble)
library(caret)
library(nnet)
library(gbm)
library(pROC)
library(kohonen)

# Get the Breast Cancer Data & create training and validation set  
data("BreastCancer") 
BC <- BreastCancer # Shortern the name of the data frame
inTrain <- sample(1:nrow(BC),2/3*nrow(BC)) # Create a 2/3 Train and 1/3 for OOT 
BC_Train <- BC[inTrain,-1]
BC_Test <- BC[-inTrain,-1]
levels(BC_Train$Class) <- c(0, 1) # recode the variables as 0 & 1 for benign and malignant 
levels(BC_Test$Class) <- c(0, 1) # recode the variables as 0 & 1 for benign and malignant 
# Take out any rows which have N/A
Find_NA_Row <- apply(BC_Train, 1, function(x){any(is.na(x))})
BC_Train <- BC_Train[!Find_NA_Row,]

Tmp <- data.frame(apply(BC_Train[1:9],2, as.numeric ))
BC_T <- cbind(Tmp, BC_Train$Class)

names(BC_T)[10] <- "Class" # rename change DV to Class  


# To develop classes from sample data based on input vectors, we are going to create self organizing maps (SOM). These classes will then be used to partition the input vectors into 2 data sets, a. train,  c. test.  

# create the color palette first for the SOM visualization 
coolBlueHotRed <- function(n, alpha = 1) {
  rainbow(n, end=4/6, alpha=alpha)[n:1]}

# create a 9 class SOM. That seems to have sufficient samples of the input vectors per class. The discrimination across the class is also good.    

require(kohonen)
SOM_Grid <- somgrid(xdim=3, ydim=3, topo="hexagonal")
SOM_Data <- as.matrix(scale(x = BC_T[,1:9]))
SOM_Model <- som(SOM_Data, grid=SOM_Grid, rlen=300,  alpha=c(0.05,0.01), keep.data= TRUE)
plot(SOM_Model, main = "Cancer Features Pattern overlaid on SOM")
plot(SOM_Model, type = "quality")
plot(SOM_Model, type="count")
plot(SOM_Model, type="dist.neighbours")

# Plot the variables to see how they distribute across the SOM's classes 
par(mfrow=c(4,2))
for ( var in 1:8 ) {  
  var_unscaled <- aggregate(as.numeric(SOM_Data[,var]), by=list(SOM_Model$unit.classif), FUN=mean, simplify=TRUE)[,2]
  plot(SOM_Model, type = "property", property=var_unscaled, main=colnames(SOM_Model$data)[var], palette.name=coolBlueHotRed)
}

# Tag the clusters to the inputs vectors in preparation for sampling for the train 
BC_T <- cbind(BC_T, SOM_Model$unit.classif)
# Rename the tagged column to "Cluster" 
names(BC_T)[11] <- "Cluster"

#Subset the vectors by their Clusters, use assign function smartly to optimize the creation of the data sets   

for (i in 1: 9){
  varname <- paste0("CL",i,"Sample")
  assign(varname, BC_T[BC_T$Cluster == i,])
    }


# Create stratified samples for each of the nine clusters, we are going to sample with replacement for 40 vectors from each cluster. Then overall, we will have 360 samples. 

for (i in 1: 9){
  varname <- paste0("CL",i,"Sample_Index")
  assign(varname, sample(nrow(get(paste0("CL",i,"Sample"))), size = 40, replace = TRUE))
}

# We are ready, create the dataset for training for the ANN model which will be boosted!  
ANN_Dataset = data.frame()
ANN_Dataset <- rbind(CL1Sample[CL1Sample_Index,],
                     CL2Sample[CL2Sample_Index,],
                     CL3Sample[CL3Sample_Index,],
                     CL4Sample[CL4Sample_Index,],
                     CL5Sample[CL5Sample_Index,],
                     CL6Sample[CL6Sample_Index,],
                     CL7Sample[CL7Sample_Index,],
                     CL8Sample[CL8Sample_Index,],
                     CL9Sample[CL9Sample_Index,])

# Create a base ANN classifier architecture using MLP ANN from RSNNS to set the "base" model to compare to the boosted model 

library(RSNNS)
require(RSNNS)

Values <- BC_T[,1:9]
Targets <- decodeClassLabels(BC_T[,10])


ANN_Train <- splitForTrainingAndTest(Values, Targets, ratio=0.15)
#ANN_Train <- normTrainingAndTestSet(ANN_Train)

ANN_Model <- mlp(ANN_Train$inputsTrain, ANN_Train$targetsTrain, 
                 size = c(8), maxit = 80, 
                 initFunc = "Randomize_Weights",initFuncParams = c(-0.3, 0.3),
                 learnFunc = "Std_Backpropagation", learnFuncParams = c(0.01,0.0),
                  updateFunc = "Topological_Order", updateFuncParams = c(0),
                  hiddenActFunc = "Act_Logistic", shufflePatterns = TRUE, 
                 linOut = FALSE,inputsTest = ANN_Train$inputsTest, targetsTest = ANN_Train$targetsTest, 
                 pruneFunc = NULL, pruneFuncParams = NULL)

par(mfrow=c(1,1))
plotIterativeError(ANN_Model)

Predict_BC <- predict(ANN_Model,ANN_Train$inputsTest)

confusionMatrix(ANN_Train$targetsTrain,fitted.values(ANN_Model))
confusionMatrix(ANN_Train$targetsTest,Predict_BC)

# See how the model performs on the holdout dataset, called BC_Test here. 
# First lets pre-process the test data before prediction 
Find_NA_Row <- apply(BC_Test, 1, function(x){any(is.na(x))})
BC_Test <- BC_Test[!Find_NA_Row,]

Tmp <- data.frame(apply(BC_Test[1:9],2, as.numeric ))
BC_Test <- cbind(Tmp, BC_Test$Class)

names(BC_Test)[10] <- "Class" # rename change DV to Class  


Test_Values <- BC_Test[,1:9]
Test_Targets <- decodeClassLabels(BC_Test[,10])
Predict_Test <- predict(ANN_Model,Test_Values)
CM_Base_Model <- confusionMatrix(Test_Targets,Predict_Test)

# Ok lets build the boosting algorithm now to improve performance. First set the key parameters for the boosting process 

Boost_iteration = 8 # The number of iterations for the boosting algorithm to generate the boosted ANN. 
B = nrow(ANN_Dataset)
RMSE_Train = vector(mode = "numeric", length = Boost_iteration) # measure model RMSE for Train at each iteration 
RMSE_Test = vector(mode = "numeric", length = Boost_iteration) # measure model RMSE for Test at each iteration 
Class_Rate = vector(mode = "numeric", length = Boost_iteration) # measure the classification rate of each iteration 

Error_T = vector(mode = "numeric", length = Boost_iteration) # Error at each iteration 
Beta_T = vector(mode = "numeric", length = Boost_iteration) # Beta at each iteration 
P = vector(mode = "numeric", length = nrow(ANN_Dataset)) # Set up the vector for sampling the input patterns in the ANN_Dataset 
D = matrix(data = rep(0, B*2), nrow = B, ncol = 2) # Set up the D matrix with rows = number of training examples and columns = 2 (for output of 1 & 0). Note we compute the P samplng distribution from the D matrix. More on this later.  

# Initialize the datasets for train and test 
Values <- ANN_Dataset[,1:9]
Targets <- decodeClassLabels(ANN_Dataset[,10])
Find_NA_Row <- apply(BC_Test, 1, function(x){any(is.na(x))})
BC_Test <- BC_Test[!Find_NA_Row,]
Test_Values <- BC_Test[,1:9]
Test_Targets <- decodeClassLabels(BC_Test[,10])

# Create placeholders to hold the boosted interation results for train and test  
Model_Output_0 = matrix(data = rep(0,B*Boost_iteration), nrow = B, ncol = Boost_iteration)
Model_Output_1 = matrix(data = rep(0,B*Boost_iteration), nrow = B, ncol = Boost_iteration)
Test_Model_Output_0 = matrix(data = rep(0,nrow(Test_Values)*Boost_iteration), nrow = nrow(Test_Values), ncol = Boost_iteration)
Test_Model_Output_1 = matrix(data = rep(0,nrow(Test_Values)*Boost_iteration), nrow = nrow(Test_Values), ncol = Boost_iteration)


# Initialize the D matrix & Sampling vector before starting iterations 
for (i in 1: nrow(ANN_Dataset)){
    D[i,which.min(Targets[i,])] <- 1/B
    }

P  <- sapply(P, function (x) x <- 1/B) # Initialize the sampling matrix with initial probabilities (1/B) 

# Function to calculate the pseudo loss value per iteration 

Calculate_Error_Iteration <- function(Predicted_Values, D, Actual_Values) {
  Error = 0 
  for (i in 1:nrow(Actual_Values)) {
    Error <- Error + 1/2*sum(D[i,])*(1-Predicted_Values[i,which.max(Actual_Values[i,])] + Predicted_Values[i,which.min(Actual_Values[i,])])
  }
  return(Error)
}

# Function to upodate the D matrix in each iteration 
UpdateD <- function(D,Beta_T,T,Predicted_Values,Actual_Values) {
  for (i in 1:nrow(D)) {
    D[i, which.min(Actual_Values[i,])] <- D[i, which.min(Actual_Values[i,])] * Beta_T[T] ^(1/2*(1+Predicted_Values[i,which.max(Actual_Values[i,])] - Predicted_Values[i,which.min(Actual_Values[i,])]))   
  }
  return(D)
}

# Update the probability distribution of sample for model post iteration based on D matrix 
Update_P <- function(Position, D) {
  return(sum(D[Position, ])/sum(D))
}

# Calculate the RMSE 

Calculate_RMSE <- function(Actual_TrainSet, Predict_TrainSet) {
  RMSE = 0 
  SSE = 0 
  for (i in 1:nrow(Actual_TrainSet)) {
    SSE = SSE + (Actual_TrainSet[i,which.max(Actual_TrainSet[i,])] - Predict_TrainSet[i,which.max(Actual_TrainSet[i,])])^2}
  
  return(RMSE <- sqrt(SSE/nrow(Actual_TrainSet)))
}


# Start the boosting process 
T = 1 

while(T<=8)
{  
  Model_Sample <- sample(c(1:B), size = B, prob = P, replace = TRUE) # Sample based on probability P. In every interation the P for each pattern "i" will be updated based on mis-classed examples. 
  
  # Develop the Input features & target outcome for the ANN model to be trained. 
  Values <- ANN_Dataset[Model_Sample,1:9]
  row.names(Values) <- 1:B
  Targets <- decodeClassLabels(ANN_Dataset[Model_Sample,10])
  Test_Targets <- decodeClassLabels(BC_Test[,10])
  
  ANN_Train <- splitForTrainingAndTest(Values, Targets, ratio=0.0)
  #ANN_Train <- normTrainingAndTestSet(ANN_Train)
  
  ANN_Model <- mlp(ANN_Train$inputsTrain, ANN_Train$targetsTrain, 
                   size = c(8), maxit = 75, 
                   initFunc = "Randomize_Weights",initFuncParams = c(-0.3, 0.3),
                   learnFunc = "Std_Backpropagation", learnFuncParams = c(0.01,0.0),
                   updateFunc = "Topological_Order", updateFuncParams = c(0),
                   hiddenActFunc = "Act_Logistic", shufflePatterns = TRUE, 
                   linOut = FALSE, inputsTest = NULL, targetsTest = NULL,
                   pruneFunc = NULL, pruneFuncParams = NULL)
  

  # Now predict the classes for 0 and 1 on the full train set 
  Predict_TrainSet <- predict(ANN_Model,ANN_Dataset[,1:9])
  row.names(Predict_TrainSet) <- 1:B
  Actual_TrainSet <- decodeClassLabels(ANN_Dataset[,10])

  CM <- confusionMatrix(Actual_TrainSet,Predict_TrainSet)
  Class_Rate[T] <- (CM[2,2] + CM[1,1])/sum(CM)
  
  #Store the model output of each iteration for the final computation of the ArgMax of the boosted model 
  Model_Output_0[,T] <- Predict_TrainSet[,1]  
  Model_Output_1[,T] <- Predict_TrainSet[,2]
  
  #Also score the test set for validation 
  Predict_TestSet <- predict(ANN_Model,Test_Values)
  row.names(Predict_TestSet) <- 1:nrow(Test_Values)
  Test_Model_Output_0[,T] <- Predict_TestSet[,1]  
  Test_Model_Output_1[,T] <- Predict_TestSet[,2]
  CM_Test <- confusionMatrix(Test_Targets,Predict_TestSet)

  # Test performance of model on training set 
  RMSE_Train[T] <- Calculate_RMSE(Actual_TrainSet, Predict_TrainSet)
  RMSE_Test[T] <- Calculate_RMSE(Test_Targets, Predict_TestSet)
  
  if (T>1) {
      if (RMSE_Test[T] > RMSE_Test[T-1]) {
        Boost_iteration <- T-1
        break
      }
  }
  
  # Now calculate the error pseudo loss function(one per iteration)
  Error_T[T] <- Calculate_Error_Iteration(Predict_TrainSet, D, Actual_TrainSet)
  # Calculate Beta based on Error function (one per iteration)
  Beta_T[T] <- Error_T[T]/(1-Error_T[T])
  # Update D matrix based on Beta and latest iteration of the model  
  D <- UpdateD(D,Beta_T,T,Predict_TrainSet,Actual_TrainSet) 
  # Update the sampling distribution based on mis-classed patterns  
  P <- sapply(1:length(P), Update_P, D)
  #P <- P/max(P)
  
  # Update to the next iteration 
  paste0("Interation # ", T, " is complete")
  T <- T + 1 
}

# Generate the final hypothesis for train set 
Boosted_Output = matrix(data = rep(0,B*2), nrow = B, ncol = 2)
for (i in 1:B) 
{
  for (j in 1:Boost_iteration){
    Boosted_Output[i,1] <- Boosted_Output[i,1] + log(1/Beta_T[j])* Model_Output_0[i,j] 
    Boosted_Output[i,2] <- Boosted_Output[i,2] + log(1/Beta_T[j])* Model_Output_1[i,j] 
    }
}

Final_Calibration <- function(x) {
  ifelse (x[1] <= x[2], return(1), return(0))    
}

Pred_Y_Boost <- apply(Boosted_Output, 1, Final_Calibration)
CM_Train <- table(Actual = ANN_Dataset[,10], Predict = Pred_Y_Boost)
paste0("Correct Classification Rate for Boosted ANN with TrainSet is... ", sprintf("%1.2f%%",100*(CM_Train[1,1]+CM_Train[2,2])/sum(CM_Train)))

# Generate the final hypothesis for test set 
Boosted_Output_Test = matrix(data = rep(0,nrow(Test_Values)*2), nrow = nrow(Test_Values), ncol = 2)
for (i in 1:nrow(Test_Values)) 
{
  for (j in 1:Boost_iteration){
    Boosted_Output_Test[i,1] <- Boosted_Output_Test[i,1] + log(1/Beta_T[j])* Test_Model_Output_0[i,j] 
    Boosted_Output_Test[i,2] <- Boosted_Output_Test[i,2] + log(1/Beta_T[j])* Test_Model_Output_1[i,j] 
  }
}

Pred_Y_Boost_Test <- apply(Boosted_Output_Test, 1, Final_Calibration)
CM_Test <- table(Actual = BC_Test[,10], Predict = Pred_Y_Boost_Test)
paste0("Correct Classification Rate for Boosted ANN with TestSet is... ", sprintf("%1.2f%%",100*(CM_Test[1,1]+CM_Test[2,2])/sum(CM_Test)))

