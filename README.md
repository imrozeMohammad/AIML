**High-level explanation of code KNN:**

The K-Nearest Neighbors (KNN) algorithm is a simple, non-parametric, supervised machine learning method used for both classification and regression tasks. It operates on the principle that similar data points tend to be close to each other in a feature space.

**Imports and Data Download:**

Imports necessary libraries (numpy, pandas, matplotlib, seaborn, sklearn).
Downloads the food delivery dataset using kagglehub.
Data Loading and Preprocessing:

**Loads the CSV data into a DataFrame.**
Samples 5000 records and selects 10 relevant columns.
Cleans column names for consistency.

**Class Imbalance Handling:**

Checks the distribution of the target variable (order_status).
Plots class distribution before balancing.
Converts order_status to binary: 1 for Delivered, 0 for others.
Upsamples the minority class to balance the dataset.
Plots class distribution after balancing.

**Feature Engineering:**

Converts the distance column from strings like '5km' or '<1km' to numeric values.
Handles missing values by dropping the 'rating' column and filling others with the median.

**Exploratory Data Analysis:**
Plots histograms for key numeric features.
Plots a correlation heatmap for numeric features.

**Model Training and Evaluation:**
Splits the data into training and test sets.
Trains a K-Nearest Neighbors (KNN) classifier.
Evaluates the model using accuracy, classification report, and confusion matrix.

This workflow covers the full machine learning pipeline: data acquisition, cleaning, balancing, feature engineering, visualization, model training, and evaluation.
