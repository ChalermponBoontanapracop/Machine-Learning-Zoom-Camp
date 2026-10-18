# Machine Learning Zoom Camp
## Introduction to Machine Learning
Machine Learning is a process of extracting patterns from data. In Supervised Learning, the data consists of Features, which are information about an object, and a Target, which is the value or outcome that we want to predict. The output of Machine Learning is a Model that learns patterns and relationships from the available data and uses the Features of new Data to generate Predictions. 

When working on a Machine Learning Project, we usually need to understand the problem, collect and prepare the Data, train the Model, evaluate the results, and use the Model. CRISP-DM organizes this process into six main stages: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment, and the process can be repeated to continuously improve the Project based on Feedback and what we learn from the actual results. 

During the Model Selection Process, we experiment with multiple Models and select the best one by splitting the Dataset into Training, Validation, and Test Sets, commonly 60%, 20%, and 20%. The Training Set is used to train the Models, the Validation Set is used to evaluate and compare different Models, and the Test Set is used to evaluate the final Model on Unseen Data. After selecting the best Model and its Settings, we can combine the Training and Validation Sets, retrain the selected Model using more Data, and evaluate it on the Test Set for the final evaluation.

## Machine Learning for Regression
Linear regression is a machine learning model used to predict a numerical target. A dataset contains features used as inputs and a target that we want the model to predict.

We split the dataset into a Training Set, Validation Set, and Test Set. The Training Set is used to train the model, the Validation Set is used to compare and tune models, and the Test Set is used to evaluate the final model.

The model uses the features in a linear equation. During training, it learns a weight for each feature and a bias term. To make a prediction, it multiplies each feature by its weight, adds the results together, and includes the bias.

Root Mean Squared Error (RMSE) combines the differences between predicted and actual values into one score. A lower RMSE indicates better predictions when comparing models on the same dataset and target scale.

Feature engineering creates or transforms features to represent the problem more effectively. Categorical variables can also contain useful information, but their values must be converted into numerical features before linear regression can use them.

Regularization helps stabilize the model by controlling the size of its learned weights. Its strength is controlled by a parameter called r.

Model tuning tests different values of r and compares their RMSE scores on the Validation Set. After selecting a value, we train the final model using the Training and Validation Sets, then evaluate it on the Test Set.

## Machine Learning for Classification
Classification is a Supervised Learning task that uses Features to predict which category an observation belongs to. In Binary Classification, the Target represents two possible categories, usually encoded as 0 and 1. Logistic Regression combines the Features with learned Weights and a Bias to calculate a Linear Score, then applies the Sigmoid Function to convert that score into an estimated Probability between 0 and 1. 

A Threshold is used to turn this Probability into a predicted Class. The Probability and the predicted Class are therefore two different outputs. Before training the Model, we prepare and explore the Data to understand missing values, data types, and the distribution of the Target. 

Risk compares the proportion of Class 1 within a group with the overall rate. Mutual Information helps us assess the relationship between a Categorical Feature and the Target, while Correlation examines linear relationships involving Numerical Features. One-Hot Encoding converts categories into binary columns without giving them an artificial order. The same preparation rules and Feature structure must be applied consistently to Training, Validation, Test, and new Data. 

During training, Logistic Regression learns the Weights and Bias by minimizing Log Loss, which measures how well the estimated Probabilities match the actual Targets and gives larger penalties to confident but incorrect Predictions. The Training Set is used to learn these patterns, the Validation Set is used to compare Models and Settings, and the Test Set is reserved for the final evaluation on Unseen Data. 

Accuracy measures the proportion of correct Class Predictions after applying a Threshold, but it should be interpreted alongside the balance between the Classes. Once the Model and its Settings are selected, we can combine the Training and Validation Sets, retrain the Model, and evaluate it on the Test Set before using it to predict new observations.

## Evaluation Metrics for Classification
After training a Logistic Regression model, evaluation metrics compare predictions with actual outcomes to assess performance. Each metric captures a different aspect of prediction quality. 

Accuracy measures the proportion of correct predictions but can be misleading when classes are imbalanced. A confusion matrix separates predictions into True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN). Precision measures how many predicted positives are correct: TP / (TP + FP). 

Recall measures how many actual positives are detected: TP / (TP + FN). F1 score combines precision and recall using their harmonic meaning. All three depend on the decision threshold. 

A ROC curve plots the True Positive Rate against the False Positive Rate across thresholds. ROC AUC measures how well the model ranks positives above negatives, with 0.5 indicating random ranking and 1.0 indicating perfect separation. 

Cross-validation evaluates performance across multiple data splits. The mean and standard deviation summarize performance and its variation, helping select hyperparameters such as C. A separate test set is reserved for final evaluation. 

## Deploying Machine Learning Models
Model Deployment is the process of taking a trained Machine Learning Model and putting it into practical use so that it can make predictions on new data without having to rerun the Training Code every time.
Once the Model has been deployed to a Server, we can create an Endpoint or API that allows other Machines or Applications to connect to the Service, send data, and receive Prediction results in return.
Model Deployment is important when we want a Model to be used by multiple Machines or Applications without requiring each system to contain its own Model Training Code.

Main Steps in Model Deployment
1. Train and Save the Model
   After training the Model, we save it as a file so that it can be loaded and used for future Predictions without having to train the Model again every time.
2. Create API Endpoints
   Create an API Endpoint that receives data from other Applications and returns Prediction results. Flask can be used as a Framework for building the Web Service.
3. Additional Options for Server Deployment
   - Pipenv: Used to create an isolated Environment for managing the Python Dependencies of the Web Service. This helps prevent the Libraries or Packages used by one Service from interfering with other Services running on the same Machine.
   - Docker: Used to package the Service inside a Docker Container together with the required System Dependencies and Python Dependencies. This allows the Service to run consistently across different Environments and reduces problems caused by differences between Machines.
4. Deploy to the Cloud
   Deploy the Docker Container to a Cloud platform such as AWS, allowing the Model Service to be accessed outside the Local Computer and enabling other Applications to call the Service through a Network or the Internet.
Therefore, Model Deployment is the process of transforming a Machine Learning Model from something that exists inside a Notebook into a Web Service that other Applications or systems can call to obtain Predictions.

