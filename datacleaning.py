import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to display the data summary
def display_data_summary(df):
    st.write("### Data Summary")
    
    st.write("**Data Types:**")
    st.write(df.dtypes)
    
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    
    st.write("**Inconsistent Values:**")
    # Check for numeric inconsistencies (you can customize this logic)
    inconsistent_values = {}
    for column in df.select_dtypes(include=['float64', 'int64']):
        # Example logic: Check for negative values in a column where it shouldn't be
        if (df[column] < 0).any():
            inconsistent_values[column] = (df[column] < 0).sum()
    st.write(inconsistent_values)

# Function to clean the data
def clean_data(df, remove_missing, impute_method, default_value):
    if remove_missing:
        df = df.dropna()
        
    # Imputation for numerical columns only
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if impute_method == 'Mean':
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    elif impute_method == 'Median':
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    elif impute_method == 'Mode':
        for col in numerical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Replace inconsistent values with the default_value
    for column in df.select_dtypes(include=['object']):
        df[column] = df[column].replace(to_replace='.*', value=default_value, regex=True)
    
    # Handle numeric inconsistencies (e.g., negative values)
    for column in numerical_cols:
        df[column] = df[column].where(df[column] >= 0, default_value)  # Replace negative values with the default
    
    return df

# Function to plot distributions
def plot_distributions(df):
    st.write("### Visualizations")
    
    for column in df.select_dtypes(include=['float64', 'int64']):
        st.write(f"**Distribution of {column}:**")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df[column], kde=True, ax=ax[0], color='blue')
        ax[0].set_title(f'Histogram of {column}')
        
        sns.boxplot(x=df[column], ax=ax[1], color='lightblue')
        ax[1].set_title(f'Box Plot of {column}')
        
        plt.show()
        st.pyplot(fig)

# Streamlit app starts here
st.title("Machine Learning Dashboard")

# Data Cleaning Tab
st.header("Data Cleaning")
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Uploaded Data")
    st.write(df.head())  # Display the first few rows of the data
    
    # Display data summary
    display_data_summary(df)
    
    # Cleaning Options
    st.write("### Data Cleaning Options")
    remove_missing = st.checkbox("Remove rows with missing values")
    
    impute_method = st.selectbox("Select imputation method for missing values:", ['None', 'Mean', 'Median', 'Mode'])
    
    default_value = st.text_input("Enter a default value for inconsistent entries (if applicable):", "0")
    
    if st.button("Clean Data"):
        cleaned_data = clean_data(df, remove_missing, impute_method, default_value)
        st.write("### Cleaned Data")
        st.write(cleaned_data)
        
        # Plot distributions for the cleaned data
        plot_distributions(cleaned_data)
