import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to display summary statistics
def display_summary_statistics(df):
    st.write("### Summary Statistics")
    st.write(df.describe())

# Function to plot distributions and visualizations
def plot_distributions(df):
    st.write("### Visualizations")
    
    # Histogram and Box Plot
    for column in df.select_dtypes(include=['float64', 'int64']):
        st.write(f"**Distribution of {column}:**")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df[column], kde=True, ax=ax[0], color='blue')
        ax[0].set_title(f'Histogram of {column}')
        
        sns.boxplot(x=df[column], ax=ax[1], color='lightblue')
        ax[1].set_title(f'Box Plot of {column}')
        
        plt.show()
        st.pyplot(fig)

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Check if there are any numerical columns
    if numerical_df.empty:
        st.write("No numerical columns available for correlation heatmap.")
        return
    
    # Compute correlation
    corr = numerical_df.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    st.pyplot()

# Function to plot pairplot
def plot_pairplot(df):
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Check if there are any numerical columns
    if numerical_df.empty:
        st.write("No numerical columns available for pairplot.")
        return
    
    # Plot pairplot
    st.write("### Pairplot")
    sns.pairplot(numerical_df)
    st.pyplot()

# Streamlit app for Data Visualization Tab
st.header("Data Visualization")
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Uploaded Data")
    st.write(df.head())  # Display the first few rows of the data
    
    # Display summary statistics
    display_summary_statistics(df)
    
    # Plot distributions
    plot_distributions(df)

    # Plot correlation heatmap
    plot_correlation_heatmap(df)
    # Call to plot pairplot
    plot_pairplot(df)
