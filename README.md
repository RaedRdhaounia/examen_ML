# Dataset Analysis and Machine Learning Model Deployment

This repository contains a Python script (`DS.py`) that performs data analysis on a dataset and deploys machine learning models for classification tasks.

## Overview

The Python script `DS.py` is structured as follows:

1. **Import Modules**: Importing necessary libraries and modules for data analysis and machine learning.

2. **Import Dataset**: Loading the dataset (`dataset_exam.csv`) for analysis.

3. **Explore Dataset**: Exploring the dataset to understand its structure, features, and target variable.

4. **Prepare Dataset**: Preprocessing the dataset by encoding categorical variables, handling missing values, and splitting into training and testing sets.

5. **Modeling and Evaluation**: Training machine learning models including Logistic Regression, XGBoost, Random Forest, and Support Vector Machine, with hyperparameter tuning where applicable. Evaluating models using accuracy score, classification report, and confusion matrix.

6. **Deployment**: Creating a pipeline for preprocessing and modeling, and deploying a RandomForestClassifier model.

## Usage

To run the script, make sure you have Python installed in your environment along with the required libraries listed in `requirements.txt`. Then, simply execute `DS.py` using Python.

## Dependencies

The project requires the following Python libraries, which can be installed using `pip install -r requirements.txt`:

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- xgboost

## Note 

the exam is based at the folder exam that is what you asked for and the folder use_guided_project is just to implementation the guided project as a plus

## Streamlit Web Application

The Streamlit web application allows users to interactively explore the dataset. Upon running the application (`streamlit run use_guided_projects/streamlite.py`), users can upload their own dataset (""" need more work for this using try and expect """) or use the default dataset provided (`dataset_exam.csv`). The application consists of two main sections:

`the code is done is one hour so sorry for the quality of need really care for promise next time will.`

1. **Basic Information**: Provides an overview of the dataset, including the first few rows, last few rows, concise summary, and descriptive statistics.

2. **Data Analysis**: Offers deeper insights into the dataset through visualizations and statistical analysis, including class distribution, missing values, duplicate rows, histograms, and correlation matrix.

### environment and packages :

make sure that you install the dependencies at the requirement file (`use_guided_projects/requirements.txt`) by the following command : 

1. open terminal at this directory 
2. write in your terminal `pip install -r use_guided_projects/requirements.txt`
3. after all packages installed you can run the web visualization using `streamlit run use_guided_projects/streamlite.py` [requirement file](/use_guided_projects/requirement.txt)
4. the project will open directly  at the web


Users can navigate between these sections using the sidebar menu and select specific analysis pages within each section. The web application leverages Streamlit's user-friendly interface to make data exploration intuitive and interactive.


## file structure 

exam/
│
├── data/
│   └── dataset_exam.csv
├── exam/
│   └── DS.py
├── data/
|   ├── requirements.txt
│   └── streamlite.csv
├── README.md
├── requirements.txt
└── LICENSE


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
