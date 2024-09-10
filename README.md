# GLOBAL DIGITAL COMPETITIVENESS: MACHINE LEARNING CLASSIFICATION

# 1. Introduction
The objective of this project is to classify the performance of countries in terms of digital and technological skills using various machine learning classification models. The dataset contains several features related to economic, technological, and educational indicators of digital competitiveness. We aim to create a binary classification model that predicts whether a country's digital/technological skills are performing well or not based on its position relative to the 80th percentile of the Digital/Technological skills index.

# 2. Problem Statement
The digital skills performance of countries plays a critical role in determining their competitiveness in the modern global economy. Countries that perform well in digital/technological skills often lead in innovation and economic growth. The goal of this project is to accurately classify the countries into high and low digital performance categories and compare various machine learning models for this task.

# 3. Data Overview
The dataset used for this project is the "DIGITAL COMPETITIVENESS-PCA.csv" file, which contains multiple variables such as:
* Foreign highly skilled personnel
* Total expenditure on R&D (%)
* Scientific research legislation
* Mobile broadband subscribers
* Skilled labor
* Health infrastructure
* GDP (Gross Domestic Product)

## Target Variable:
We engineered the target variable, Digital skills Performance, by categorizing countries into two groups:
* High performers: Countries with digital/technological skills above the 80th percentile.
* Low performers: Countries with digital/technological skills below the 80th percentile.

# 4. Methodology
## 4.1 Data Preprocessing
The data preprocessing steps include:
* Loading the dataset.
* Creating the target variable Digital skills Performance using the 80th percentile of the Digital/Technological skills column. Countries above the threshold are classified as high performers (value = 1), while those below are classified as low performers (value = 0).
* Dropping the Digital/Technological skills column after creating the target variable.
* Splitting the data into training (80%) and testing (20%) sets for model evaluation.

## 4.2 Machine Learning Models
We applied various machine learning models to classify the target variable. The models used are:
* Logistic Regression
* Decision Trees
* Random Forest
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Gradient Boosting Machines (GBM)
Each model was trained using the training dataset and then evaluated on the test dataset. The performance was assessed using a variety of evaluation metrics, including accuracy, precision, recall, and F1-score.

## 4.3 Model Evaluation Metrics
The following evaluation metrics were used to compare model performance:
* Accuracy: The percentage of correct predictions out of all predictions.
* Precision: The ability of the classifier to correctly identify only relevant instances.
* Recall: The ability of the classifier to find all the relevant cases within the dataset.
* F1-Score: A weighted average of precision and recall.
* Confusion Matrix: Provides insight into the number of true positives, true negatives, false positives, and false negatives.

# 5. Results
## 5.1 Model Performance Metrics
The performance of each model is presented in terms of accuracy, precision, recall, and F1-score. Below are the key metrics for each classifier:

![MODEL EVALUATION METRICS](https://github.com/user-attachments/assets/16cc34d7-d673-4557-8f99-0269fcf48ca7)

As can be seen above, the random forest model out performs all the other models on all the mentioned metrics.

## 5.2 Confusion Matrix Analysis
We also visualized the confusion matrix for each model to better understand the distribution of predictions. The confusion matrix for each classifier showed the following breakdown:
* True Positives (TP): Correctly predicted high performers.
* True Negatives (TN): Correctly predicted low performers.
* False Positives (FP): Low performers incorrectly predicted as high performers.
* False Negatives (FN): High performers incorrectly predicted as low performers.
These insights were represented in heatmaps for better visualization.

# 6. Discussion
Best-Performing Model: The random forest model achieved the highest accuracy of 91%, followed closely by K-nearest neighbor (kNN) with 90%. Both ensemble methods outperformed the other models, indicating that they capture the complexities of the data better.

# 7. Conclusion
In this project, we successfully implemented and evaluated several machine learning models to classify digital skills performance based on a country's technological, economic, and educational indicators. The Gradient Boosting Machines model was the most effective in classifying high and low performers in digital/technological skills. The results demonstrate that ensemble models (GBM and Random Forest) perform significantly better than simpler models such as Logistic Regression or Naive Bayes in this context.

# Future work
Future improvements could include:
* Fine-tuning hyperparameters for each model.
* Model deployment to make predictions.
* Investigating the effect of feature scaling on models like KNN.
* Exploring additional ensemble methods such as XGBoost or stacking models for improved performance.
* This project illustrates the application of machine learning techniques in the context of national digital competitiveness, providing valuable insights for stakeholders looking to understand and enhance digital skills performance globally.
