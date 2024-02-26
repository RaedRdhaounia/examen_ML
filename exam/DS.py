import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import dataset
"""
    Read dataset from CSV file

    Arguments:
    - file_path: str, path to the CSV file

    Returns:
    - dataset: DataFrame, loaded dataset
"""
dataset = pd.read_csv('data/dataset_exam.csv')

# Explore dataset

# Display first few rows of the dataset
"""
    Display first few rows of the dataset
"""
print("First few rows of the dataset:")
print(dataset.head())

# Display last few rows of the dataset
"""
    Display last few rows of the dataset
"""
print("\n Last few rows of the dataset:")
print(dataset.tail())

# Display concise summary of the dataset
"""
    Display concise summary of the dataset
"""
print("\n Concise summary of the dataset:")
print(dataset.info)

# Generate descriptive statistics of the dataset
"""
    Generate descriptive statistics of the dataset
"""
print("\n Descriptive statistics of the dataset:")
print(dataset.describe())

# Count occurrences of each class in the 'Class' column
"""
    Count occurrences of each class in the 'Class' column
"""
print("\n Occurrences of each class in the 'Class' column:")
print(dataset['Class'].value_counts())

# Check for missing values in the dataset and visualize using heatmap
"""
    Check for missing values in the dataset and visualize using heatmap
"""
print("\n Missing values in the dataset:")
print(dataset.isnull().sum())
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap="Blues")

# Count duplicate rows in the dataset
"""
    Count duplicate rows in the dataset
"""
print("\n Duplicate rows in the dataset:", dataset.duplicated().sum())

# Visualize histogram for each numerical feature
"""
    Visualize histogram for each numerical feature
"""
dataset.hist(bins=30, figsize=(20, 20), color='r')
plt.show()

# Compute pairwise correlation of columns
"""
    Compute pairwise correlation of columns
"""

# Replace non-numeric values with NaN and then fill NaN with the mean of the column
# Identify non-numeric columns
non_numeric_columns = dataset.select_dtypes(exclude=[np.number]).columns

# Exclude non-numeric columns when computing the correlation matrix
numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation_matrix = numeric_dataset.corr()

# Print the correlation matrix
print("\nPairwise correlation of numeric columns:")
print(correlation_matrix)

# Plot pairwise relationships in the dataset
"""
    Plot pairwise relationships in the dataset
"""
sns.pairplot(dataset, height=2.5)
plt.show()

# Plot correlation matrix using heatmap
"""
    Plot correlation matrix using heatmap
"""
# Identify non-numeric columns
non_numeric_columns = dataset.select_dtypes(exclude=[np.number]).columns

# Exclude non-numeric columns when plotting heatmap
numeric_dataset = dataset.drop(columns=non_numeric_columns)

# Plot correlation matrix using heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(numeric_dataset.corr(), annot=True)
plt.show()

# Select specific features for analysis
"""
    Select specific features for analysis
    
    Arguments:
    - dataset: DataFrame, input dataset
    - features: list, list of feature names
    
    Returns:
    - selected_features: DataFrame, selected features
"""
selected_features = dataset[['F3', 'F8', 'F11', 'F15']]

# Create box plots for each selected feature
"""
    Create box plots for each selected feature
    
    Arguments:
    - selected_features: DataFrame, selected features
"""
plt.figure(figsize=(12, 8))
sns.boxplot(data=selected_features)
plt.title('Box Plot of F3, F8, F11, F15')
plt.show()

# Function to identify outliers in a column
"""
    Function to identify outliers in a column
    
    Arguments:
    - column: Series, input column
    
    Returns:
    - outliers: Series, boolean mask indicating outliers
"""
def identify_outliers(column):
    Q1 = column.quantile(0.25)  # Compute the first quartile
    Q3 = column.quantile(0.75)  # Compute the third quartile
    IQR = Q3 - Q1  # Compute the interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Compute the lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Compute the upper bound for outliers
    outliers = (column < lower_bound) | (column > upper_bound)  # Identify outliers
    return outliers

# Identify outliers in each selected feature
"""
    Identify outliers in each selected feature
    
    Arguments:
    - dataset: DataFrame, input dataset
"""
outliers_f3 = identify_outliers(dataset['F3'])
outliers_f8 = identify_outliers(dataset['F8'])
outliers_f11 = identify_outliers(dataset['F11'])
outliers_f15 = identify_outliers(dataset['F15'])

# Print the indices of outliers
"""
    Print the indices of outliers
"""
print("\nIndices of outliers in F3:", dataset.index[outliers_f3])
print("Indices of outliers in F8:", dataset.index[outliers_f8])
print("Indices of outliers in F11:", dataset.index[outliers_f11])
print("Indices of outliers in F15:", dataset.index[outliers_f15])

# Prepare dataset

# Encode categorical variables using LabelEncoder
"""
    Encode categorical variables using LabelEncoder
"""
label_encoder = LabelEncoder()
for col in ['F1', 'F4', 'F5', 'F6', 'F7', 'F9', 'F10', 'F11', 'F12', 'F13', 'Class']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

print(dataset['Class'].value_counts())

# Convert dataset to numeric type and handle missing values by filling with mean
"""
    Convert dataset to numeric type and handle missing values by filling with mean
"""
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset.fillna(dataset.mean(), inplace=True)

# Separate features (X) and target variable (y)
"""
    Separate features (X) and target variable (y)
"""
X = dataset.drop('Class', axis=1)
y = dataset['Class']

# Split the data into training and testing sets
"""
    Split the data into training and testing sets
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
"""
    Standardize numerical features
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeling and evaluation

# Train Logistic Regression with hyperparameter tuning
"""
    Train Logistic Regression with hyperparameter tuning
"""
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
model = LogisticRegression()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test_scaled, y_test)
print(f"\nAccuracy on the test set (Logistic Regression): {accuracy}")

# Train XGBoost Classifier
"""
    Train XGBoost Classifier
"""
model = XGBClassifier()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (XGBoost): {accuracy:.2f}')
print('\n XGBoost Classification Report:')
print(classification_report(y_test, y_pred))
print('\n XGBoost Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Train Random Forest Classifier
"""
    Train Random Forest Classifier
"""
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (Random Forest): {accuracy:.2f}')
print('\n Random Forest Classification Report:')
print(classification_report(y_test, y_pred))
print('\n Random Forest Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Train SVM model
"""
    Train SVM model
"""
model = SVC(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (SVM): {accuracy:.2f}')
print('\n SVM Classification Report:')
print(classification_report(y_test, y_pred))
print('\n SVM Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Deployment

# Create a pipeline for preprocessing and modeling
"""
    Create a pipeline for preprocessing and modeling
"""
num_features = X_train.select_dtypes(include=[np.number]).columns
cat_features = X_train.select_dtypes(include=[object]).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('transformer', PowerTransformer(method='yeo-johnson', standardize=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, num_features),
        ('categorical', SimpleImputer(strategy='most_frequent'), cat_features)
    ])

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the pipeline
"""
    Train the pipeline
"""
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
full_pipeline.fit(X_train, y_train)

# Make predictions on the test set
"""
    Make predictions on the test set
"""
y_pred = full_pipeline.predict(X_test)

# Evaluate the model
"""
    Evaluate the model
"""
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (Pipeline): {accuracy:.2f}')
print('\n MODEL Classification Report:')
print(classification_report(y_test, y_pred))
print('\n MODEL Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
