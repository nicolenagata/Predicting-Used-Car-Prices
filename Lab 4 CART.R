# 1. Importing Libraries -------------------------------------------------------
library(rpart)
library(rpart.plot)
library(forecast)
library(caret)
library(ROSE)
library(treeClust)
library(dplyr)

# 1.2 Load Data ----------------------------------------------------------------
toyota <- read.csv("ToyotaCorolla.csv", header = TRUE)

# Data Exploration
head(toyota, 10)
str(toyota)
names(toyota)
nrow(toyota)

# just set the target variable as a factor, 
# leave the predictor variables as is
toyota$Fuel_Type <- as.factor(toyota$Fuel_Type)

# 1.3 Training validation split -------------------------------------------------------

set.seed(666)

train_index <- sample(1:nrow(toyota), 0.6 * nrow(toyota))
valid_index <- setdiff(1:nrow(toyota), train_index)

train_df <- toyota[train_index, ]
valid_df <- toyota[valid_index, ]

# checking just to be sure
nrow(train_df)
nrow(valid_df)

head(train_df)
head(valid_df)

str(train_df)
str(valid_df)

# 2. Regression Tree -----------------------------------------------------------

regress_tr <- rpart(Price ~ Age_08_04 + KM + Fuel_Type + HP + Automatic + Doors
                    + Quarterly_Tax + Mfr_Guarantee + Guarantee_Period + Airco +
                    Automatic_airco + CD_Player + Powered_Windows + Sport_Model
                    + Tow_Bar, data = train_df, method = "anova", maxdepth = 3)

rpart.plot(regress_tr, type = 4)

" The resulting regression tree with a depth of 3, classifies the price of used
Toyota Corolla cars based on a few predictors. In this regression tree, the
main predictors are most likely Age and HP, as these variables appear at the
higher splits. "

# Predicting Training Set 
predict_train <- predict(regress_tr, train_df)
accuracy(predict_train, train_df$Price)
sd(train_df$Price)

# Predicting Validation Set
predict_valid <- predict(regress_tr, valid_df)
accuracy(predict_valid, valid_df$Price)
sd(valid_df$Price)

"The RMSE of the training set is 1,350.029.
The RMSE of the valdiation set is 1,500.804.

Since the errors (RMSE) of both the training and validation sets are relatively 
close together and lower than their respective standard deviations to Price,
the predictors shown in the regression tree pretty accurately 
classifies the price of used Totyota Corollas. It is expected for the
validation set error to be slightly higher than training, so this model 
produces a realistic and good quality regression tree.
"

# 3. Predict new record using regression tree ----------------------------------

# df of new records
new_record <- data.frame(Age_08_04 = 77, 
                         KM = 117000, 
                         Fuel_Type = "Petrol", 
                         HP = 110, 
                         Automatic = 0, 
                         Doors = 5, 
                         Quarterly_Tax = 100, 
                         Mfr_Guarantee = 0, 
                         Guarantee_Period = 3, 
                         Airco = 1, 
                         Automatic_airco = 0, 
                         CD_Player = 0, 
                         Powered_Windows = 0, 
                         Sport_Model = 0, 
                         Tow_Bar = 1)

# predict the price of new record
regress_tr_pred <- predict(regress_tr, newdata = new_record)
regress_tr_pred

"The predicted price of the new record's Toyota Corolla will be $7,969.59"

# Range

  # calculating which node the new prediction falls under
new_record_node <- rpart.predict.leaves(regress_tr, newdata = new_record,
                                       type = "where")
new_record_node
  # df with new record, node, prediction
new_record_pred <- data.frame(Node = new_record_node,
                             Prediction = regress_tr_pred)
new_record_pred
  # associating each record from training set to a node
which_node_train <- rpart.predict.leaves(regress_tr, newdata = train_df,
                                         type = "where")
head(which_node_train)
  # sd of each terminal node
sd_node = aggregate(train_df$Price, list(which_node_train), FUN = sd)
names(sd_node) <- c("Node", "sd")
sd_node
  # min of each terminal node
min_node = aggregate(train_df$Price, list(which_node_train), FUN = min)
names(min_node) <- c("Node", "min")
min_node
  # max of each terminal node
max_node = aggregate(train_df$Price, list(which_node_train), FUN = max)
names(max_node) <- c("Node", "max")
max_node
  # df of new record, node, prediction, min, max, sd
new_record_pred_range <- new_record_pred %>%
  inner_join(min_node, by = "Node") %>%
  inner_join(max_node, by = "Node") %>%
  inner_join(sd_node, by = "Node")

new_record_pred_range

" The new record falls under node 4, with a price of $7,969.59.
The minimum price for node 4 is $4,400.
The maximum price for node 4 is $10,500
The standard deviation for node 4 is $1,028.88."

# 4. Classification tree --------------------------------------------------

# Converting Price Variable to categorical
toyota$cat_price <- ifelse(toyota$Price <= mean(toyota$Price, na.rm = TRUE), "0", "1")
table(toyota$cat_price)

toyota$cat_price <- as.factor(toyota$cat_price)

mean(toyota$Price, na.rm = TRUE)
# Remove the numerical Price variable to avoid confusion
toyota_cat <- toyota[,- c(3)]
names(toyota_cat)

# 4.1 Training validation split -------------------------------------------

set.seed(666)

train_cat_index <- sample(1:nrow(toyota_cat), 0.6 * nrow(toyota_cat))
valid_cat_index <- setdiff(1:nrow(toyota_cat), train_cat_index)

train_cat_df <- toyota_cat[train_cat_index, ]
valid_cat_df <- toyota_cat[valid_cat_index, ]

nrow(train_cat_df)
nrow(valid_cat_df)

head(train_cat_df)
head(valid_cat_df)

# 4.2 Classification tree -------------------------------------------------

class_tr <- rpart(cat_price ~ Age_08_04 + KM + Fuel_Type + HP + Automatic + Doors
                    + Quarterly_Tax + Mfr_Guarantee + Guarantee_Period + Airco +
                      Automatic_airco + CD_Player + Powered_Windows + Sport_Model
                    + Tow_Bar, data = train_cat_df, method = "class", maxdepth = 3)

rpart.plot(class_tr, type = 4)

" Top Predictors:
The resulting classification tree with a depth of 3, predicts a high or low
price of used Toyota Corolla cars based on a few predictors. In this
classifcation tree, the main predictor is Age because the predicor appears at
higher splits, however KM is also a relevant predicotr, appearing towards the
bottom split."

# 4.3 Confusion Matrix ---------------------------------------------------------

# training set
class_tr_train_predict <- predict(class_tr, train_cat_df, type = "class")

t(t(head(class_tr_train_predict,10)))

confusionMatrix(class_tr_train_predict, train_cat_df$cat_price, positive = "1")

" The classification tree model produces a pretty accurate prediction based on
high accuracy (90.01%), similar and high sensivtivity (92.26%) and specificity 
(88.75%), and high and similar Pos Pred (82.18%) and Neg Pred (95.32%)."

# validation set
class_tr_valid_predict <- predict(class_tr, valid_cat_df, type = "class")

t(t(head(class_tr_valid_predict,10)))

confusionMatrix(class_tr_valid_predict, valid_cat_df$cat_price, positive = "1")

" The classification tree model produces a pretty accurate prediction on the
validation set based on high accuracy (89.04%), similar and high sensivtivity
(87.45%) and specificity (90.12%), and high and similar Pos Pred (85.59%)
and Neg Pred (91.45%)."

# The probabilities
class_tr_valid_predict_prob <- predict(class_tr, valid_df,
                                       type = "prob")

head(class_tr_valid_predict_prob)

# ROC Curve
ROSE::roc.curve(valid_cat_df$cat_price, class_tr_valid_predict)

" The area under the ROC cruve is 0.888. Since the area under the ROC cruve
is greater than 0.7, the classification model can accurately predict high and
low classifications for Toyota Corolla car prices."

"How do the accuracies compare?
The accuriacies for the training and validation set are similar, with only a 
0.97% difference. The training set accuracy was higher, but that is expected
for the model to slightly better predict in training rather than the
validation set. Since both accuracies remain relatively high and close in
percentages, I would say the model is pretty accurate."

# 4.4 Predict new record --------------------------------------------------

class_tr_pred <- predict(class_tr, newdata = new_record)
class_tr_pred

" The new record's Toyota Corolla will be priced low."

# 5. Comparing the trees --------------------------------------------------

" Are the predictors similar?
In the regression tree, the top predictor is age and HP is also a relevant
predictor. In the classification tree, the top predictor is age and KM is also
a relevant predictor. The predictors are similar in that age is the top
predictor in both trees, only varying slightly with the less prominent 
predictors."

"Are the predictions similar?
The regression tree predicted the new record to be priced at $7,969.59.
The classification tree predicted the new record to be priced low (0).
The classification tree predicts high or low based on the mean price, being $10,730.82.
Any car priced above the mean will recieve a 1 (high),
any car priced below the mean will recieve a 0 (low).
The predictions for both the regression tree and classification tree are similar,
since $7,969.59 is less than the price mean and hence classifed as low.
"

" If you are running a business, which tree would you use? Why?
If I was running a business, I would use the regression tree because it
provides an exact car price and the tree is more in depth, with slightly more
predictors and splits than the classification tree, but still simple to
understand. The only downside to using a regression tree is having to
calculate the range of each node seperately to get a better understanding 
of the spread of data.
The classification tree is super simple to understand, but only tells you
whether a car will be priced high or low and the probability. Hence I beleive
a regression tree gives a simple, but more indepth prediction especially
for a business."












