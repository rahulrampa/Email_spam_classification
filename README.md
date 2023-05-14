# Spam vs Non-Spam Email
This project aims to develop a machine learning model that can accurately classify emails as either spam or non-spam.

## Data
The dataset used for this project contains a collection of emails labeled as either "spam" or "non-spam". The dataset can be found at [http://archive.ics.uci.edu/ml/datasets/Spambase]. File “spambase.data” contains the actual data, and files “spambase.names” and “spambase.DOCUMENTATION” contain the description of the data. This dataset has 4601 records, each record representing a different email message. Each record is described with  58  attributes  (indicated  in  the  aforementioned  .names  file):  attributes  1-57  represent  various  content-based characteristics already extracted from each email message (related to the frequency of certain words or certain punctuation symbols in a message as well as to the usage of capital letters in a message), and the last attribute represents the class label for each message (spam or non-spam).

## Requirements
The following packages are required to run the code in this project:
* numpy
* pandas
* scikit-learn
* lightgbm
* xgboost
* scikitplot
* matplotlib
* seaborn


## Evaluation
The model is evaluated using accuracy, precision, recall and F1-score metrics.

## Conclusion
1. Standard Model:
I chose the standard train-test split method for this dataset over the nested cross-validation structure because I want to utilize stacking techniques to stack all the fine-tuned base models. I tried out `Naive Bayes`, `Logistic Regression`, `KNN`, `SVM`, `Random Forest`, `LightGBM`, `Xgboost` along with the `RandomizedSearchCV` to explore the different hyperparameter combinations(including different preprocessing techniques) for all of the base models. `LightGBM` has the best predictive performance among these base models with an accuracy score and AUC over 95.5% and 98.5%, respectively. However, if we further stack these base models(exclude the fine-tuned `Logistic Regression`) with the regular `Logistic Regression` as the final meta-model, we can even improve the aforementioned metrics to over 99%. Therefore, I used this stacking classifier as the final model and applied it to the testing dataset. 
The final model demonstrates a good performance on the testing set with predictive accuracy, precision, recall, f-measure all above 95%. Moreover, the ROC (with AUC = 0.99) curve implies that the model has the ability to rank the positive class higher than the negative class consistently. The precision-recall curve is also included to show that the model can maintain approximately 95% precision until 80% recall (the decline, in the beginning, might indicate some hard predicting targets in the testing set). As for the Lift Curve, we can see that the model's top 40% prediction is almost 2.5x as good at predicting the outcome as a random guess.

2. Cost-sensitive model:
For the cost-sensitive model, I follow a similar procedure and structure in the regular model but with two main differences. First, I set the `class_weight` parameter to adjust the weight of the two classes. With the {0:10, 1:1} setting, this can tell the classifier that not spam(0) class is 10x as crucial as the spam (1) class, and they should incorporate this during the training process. Second, I built a `customized cost function` to calculate the average misclassification cost and leveraged this function for searching the best hyperparameter combinations.
Overall, this final cost-sensitive model is not comparable with the general model we build earlier in terms of predictive power. Still, it demonstrates a good general performance with accuracy, precision, recall, f-measure all above 90%. More importantly, it can make a classification that generates a much lower total cost than the general model with only minor sacrifices on the common classification performance metrics. (*cost-sensitive cost: 114* vs *general model cost: 212*, base on the confusion matrix in the plots) 
