import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the web app
st.title("Data Analysis App")
dataset_source = st.file_uploader("Upload dataset", type=["csv", "txt"])


# Button for uploading dataset
if dataset_source is not None:
    # Check if file is uploaded
    dataset = pd.read_csv(dataset_source)
else:
# If default dataset is selected or no dataset is uploaded
    dataset = pd.read_csv('data/dataset_exam.csv')

interpretation_text = """
The dataset with has 16  columns, each representing a different feature, and a class label column at the end:

- **F1**: Categorical feature indicating the type of applicant.
- **F2**: Continuous feature representing some numeric value.
- **F3**: Continuous feature representing another numeric value.
- **F4**: Categorical feature indicating a certain attribute.
- **F5**: Categorical feature indicating another attribute.
- **F6**: Categorical feature indicating a specific category.
- **F7**: Categorical feature indicating a certain category.
- **F8**: Continuous feature representing a numeric value.
- **F9**: Categorical feature indicating a certain attribute.
- **F10**: Categorical feature indicating a certain attribute.
- **F11**: Continuous feature representing a numeric value.
- **F12**: Categorical feature indicating a certain attribute.
- **F13**: Categorical feature indicating a certain attribute.
- **F14**: Continuous feature representing a numeric value.
- **F15**: Continuous feature representing another numeric value.
- **Class**: Target variable indicating the class label.

Each row in the dataset represents an observation or data point, with each column representing a specific feature or attribute associated with that observation. The interpretation may vary based on the context of the dataset and the specific domain it pertains to.
"""

# Define functions for different sections of dataset exploration
def show_first_few_rows():
    st.subheader("First few rows of the dataset:")
    st.write(dataset.head())
    # Interpretation
    st.text(interpretation_text)

def show_last_few_rows():
    st.subheader("Last few rows of the dataset:")
    st.write(dataset.tail())

    num_rows = dataset.shape[0]
    interpretation_text = f"""
    The table above displays the last few rows of the dataset, providing a glimpse of the data towards the end.
    It shows {num_rows} rows from the bottom of the dataset.
    
    The class (target variable) in the dataset has two categories: "+" and "-". This can be inferred from the fact that the first five rows had class values of "+", and the last five rows have class values of "-".
    """
    st.text(interpretation_text)

def show_concise_summary():
    st.subheader("Concise summary of the dataset:")
    st.write(dataset.info())

def show_descriptive_statistics():
    st.subheader("Descriptive statistics of the dataset:")
    descriptive_stats = dataset.describe()
    st.write(descriptive_stats)

    # Calculate the interquartile range (IQR) and differences between quartiles and extreme values
    for column in descriptive_stats.columns:
        Q1 = descriptive_stats.loc['25%', column]
        Q3 = descriptive_stats.loc['75%', column]
        IQR = Q3 - Q1
        min_value = descriptive_stats.loc['min', column]
        max_value = descriptive_stats.loc['max', column]

        min_Q1_diff = Q1 - min_value
        Q3_max_diff = max_value - Q3

        st.write(f"Feature: {column}, IQR: {IQR}, Difference between Q1 and min: {min_Q1_diff}, Difference between Q3 and max: {Q3_max_diff}")

        # Check for potential outliers
    if min_Q1_diff > 1.5 * IQR or Q3_max_diff > 1.5 * IQR:
        st.header("Potential outliers detected.")
    else:
        st.header("No potential outliers detected.")
    
    st.subheader("Analyse")
    st.write("""
The descriptive statistics provide insights into the distribution of numerical features in the dataset:

- The total number of observations in the dataset is 689.

- The average values for F3, F8, and F11 are 4.77, 2.22, and 2.40 respectively. 
  ===> These averages give us a general idea of the central tendency of the data.

- The standard deviations for F3, F8, and F11 are 4.98, 3.35, and 4.87 respectively. 
  ===> These values indicate the spread or variability of the data around the mean.

- The minimum values for F3, F8, and F11 are 0.0, 0.0, and 0.0 respectively. 
  ===> This shows the lowest values observed for each feature.

- The median values for F3, F8, and F11 are 2.75, 1.0, and 0.0 respectively. 
  ===> The median represents the middle value of the dataset and is less affected by outliers.

- The third quartile values for F3, F8, and F11 are 7.25, 2.63, and 3.0 respectively. 
  ===> These quartiles divide the dataset into four equal parts, providing insights into the spread of data.

- The maximum values for F3, F8, and F11 are 28.0, 28.5, and 67.0 respectively. 
  ===> These are the highest values observed for each feature, indicating the upper range of the dataset.
""")

    st.subheader("Interpretation")

    st.write("""
The descriptive statistics, particularly the standard deviation and range of values (minimum and maximum), provide insights into the spread of the data. Here's how each component contributes to understanding the spread:

1. Standard Deviation:
- The standard deviation measures the dispersion or spread of the values around the mean.
- A higher standard deviation indicates greater variability in the data points.
- For example, if the standard deviation is high for a particular feature like F3, it suggests that the values of F3 are spread out over a wider range from the mean.

2. Range of Values (Minimum and Maximum):
- The minimum and maximum values give the boundaries within which the data points lie.
- The difference between the maximum and minimum values represents the overall range or spread of the dataset.
- For example, if the range between the minimum and maximum values of a feature like F8 is large, it indicates that the data points for F8 are spread out over a wide range of values.

By considering both the standard deviation and the range of values, we can gain a comprehensive understanding of how spread out the data points are within each feature of the dataset. This understanding is crucial for assessing the variability and distribution of the data, which in turn informs various analytical decisions and interpretations.
""")

def show_class_distribution():
    st.subheader("Occurrences of each class in the 'Class' column:")
    st.write(dataset['Class'].value_counts())

    st.header("Target Distribution")
    st.subheader(" (' - ') ==> 383 (' + ') ==> 306")
    st.write("This distribution imbalance may affect the model's ability to generalize well, particularly if the minority class ('+') is underrepresented. Strategies like resampling techniques or using algorithms robust to imbalanced data may be needed to address this issue and improve model performance.", unsafe_allow_html=True)

def show_missing_values():
    st.subheader("Missing values in the dataset:")
    st.write(dataset.isnull().sum())

    st.header("")
    st.text(
    """
Having no missing values in the dataset is generally considered a positive aspect as it simplifies data preprocessing and analysis.
It ensures that all observations can be used for training machine learning models without the need for imputation or handling missing data, which can introduce biases or inaccuracies.
It reduces the risk of introducing errors during data processing. 
""")
    st.subheader(" The absence of missing values is beneficial for building robust and reliable machine learning models.")
def show_duplicate_rows():
    st.subheader("Duplicate rows in the dataset:")
    st.write(dataset.duplicated().sum())

    st.write(""" The output "0" indicates that there are no duplicate rows in the dataset. This means that each row in the dataset is unique, and there are no exact duplicates present.""")
    st.subheader("Having no duplicate rows is generally considered desirable, as it avoids redundancy and ensures the integrity of the data.")

def show_histograms():
    st.subheader("Visualizing histogram for each numerical feature:")
    numeric_columns = dataset.select_dtypes(include='number')
    plt.figure(figsize=(20, 20))
    numeric_columns.hist(bins=30, color='r', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)
    st.write(
"""
From the histograms, it sounds like F3 and F8 have a similar distribution, with their maximum values (90th percentile) being similar, whereas F11 and F15 also have a similar distribution, with their maximum values being significantly higher. This observation indicates that F11 and F15 may have a wider range of values compared to F3 and F8.
This insight is valuable for understanding the spread and variability of these features in your dataset. It suggests that F11 and F15 may have a larger range of values, possibly indicating a broader diversity or variability in the data captured by these features compared to F3 and F8.
""")

def show_correlation_matrix():
    st.subheader("Correlation matrix using heatmap:")
    plt.figure(figsize=(8, 8))
    numeric_columns = dataset.select_dtypes(include='number')
    correlation_matrix = numeric_columns.corr()
    sns.heatmap(correlation_matrix, annot=True)
    st.pyplot(plt)

    st.subheader("Correlation Matrix Analysis")

    st.write("### Moderate to Weak Positive Linear Relationships:")
    st.write("- The correlation coefficients between various pairs of features range from 0.64 (F15 and F11) to 0.051 (F15 and F8).")
    st.write("  - **Interpretation:** These coefficients indicate varying degrees of positive linear relationships between pairs of features. For instance, a coefficient of 0.64 suggests a moderate positive correlation, implying that as one feature increases, the other tends to increase as well. Conversely, a coefficient of 0.051 indicates a weaker positive correlation.")

    st.write("### Absence of Strong Negative Linear Relationships:")
    st.write("- No negative correlation coefficients are observed in the correlation matrix.")
    st.write("  - **Interpretation:** This suggests that there are no strong negative linear relationships between any pair of features. In other words, as one feature increases, the other does not consistently decrease.")

    st.write("### No Strong Correlations Close to 1 or -1:")
    st.write("- None of the correlation coefficients are close to 1 or -1.")
    st.write("  - **Interpretation:** The absence of coefficients near 1 or -1 indicates that there are no perfect linear relationships between any pair of features. This means that no pair of features exhibits a strong positive or negative linear dependency on each other.")

    st.write("### Presence of Moderate to Weak Positive Correlations:")
    st.write("- Several correlation coefficients fall within the range of 0.3 to 0.1.")
    st.write("  - **Interpretation:** Coefficients in this range suggest moderate to weak positive linear relationships between pairs of features. For example, a coefficient of 0.3 indicates a moderate positive correlation, while a coefficient of 0.1 suggests a weaker positive correlation.")

    st.write("### Overall Insights:")
    st.write("- The correlation matrix heatmap provides valuable insights into the strength and direction of linear relationships between pairs of numerical features in the dataset. These insights help identify potential associations that may be further explored in subsequent data analysis and modeling tasks.")

# Create a dictionary mapping page names to functions
# Divide pages into groups
basic_info_pages = {
    "First few rows": show_first_few_rows,
    "Last few rows": show_last_few_rows,
    "Concise summary": show_concise_summary,
    "Descriptive statistics": show_descriptive_statistics,
}

data_analysis_pages = {
    "Class distribution": show_class_distribution,
    "Missing values": show_missing_values,
    "Duplicate rows": show_duplicate_rows,
    "Histograms": show_histograms,
    "Correlation matrix": show_correlation_matrix
}

# Combine groups into a single dictionary
pages = {
    "Basic Information": basic_info_pages,
    "Data Analysis": data_analysis_pages
}
# Define functions to show different sections of data analysis
def show_basic_information():
    st.write("This is the Basic Information section")


def show_data_analysis():
    st.write("This is the Data Analysis section")


def show_data_overview():
    st.write("This is the Data Overview section")

# Main title
st.title("Data Analysis of the Exam")

# Sidebar title
st.sidebar.title("Data Analysis of the Exam")
# Define sidebar navigation for Basic Information and Data Analysis
selected_group = st.sidebar.radio("Navigation", ["Basic Information", "Data Analysis"])

# Define pages for Basic Information and Data Analysis
if selected_group == "Basic Information":
    selected_page = st.radio("Select Basic Information Page", list(basic_info_pages.keys()))
    basic_info_pages[selected_page]()  # Call the function corresponding to the selected page
else:  # Data Analysis
    selected_page = st.radio("Select Data Analysis Page", list(data_analysis_pages.keys()))
    data_analysis_pages[selected_page]()  # Call the function corresponding to the selected page

