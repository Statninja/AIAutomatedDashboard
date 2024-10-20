import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# CSS for futuristic design
st.markdown(
    """
    <style>
    body {
        background-color: #1c1c1c;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #444;
        border: 2px solid #00bfff;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        padding: 15px 30px;
        margin: 10px;
    }
    .stButton>button:hover {
        background-color: #00bfff;
        color: black;
    }
    .stDataFrame {
        background-color: #000;
        color: white;
        border: 1px solid #00bfff;
    }
    .stRadio>div {
        background-color: #444;
        border-radius: 10px;
        padding: 10px;
    }
    .stRadio>div:hover {
        background-color: #00bfff;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True
)

# Function to propose cleaning steps and add messages
def propose_cleaning_steps(df):
    cleaning_steps = []
    if df.isnull().values.any():
        cleaning_steps.append("Handle Missing Values")
        st.warning("The dataset contains missing values.")
    else:
        st.info("No missing values found.")
        
    if (df.dtypes == 'object').any():
        cleaning_steps.append("Fix Inconsistent Formatting")
        st.warning("The dataset contains text fields that might have inconsistent formatting.")
    else:
        st.info("No inconsistent formatting found.")
        
    if df.duplicated().any():
        cleaning_steps.append("Remove Duplicates")
        st.warning("The dataset contains duplicate rows.")
    else:
        st.info("No duplicates found.")
    
    if not cleaning_steps:
        cleaning_steps = ["No specific issues found, but you can clean the data if needed."]
    
    return cleaning_steps

# Function to clean data
def clean_data(df, selected_steps, fill_option=None):
    if selected_steps:
        if "Handle Missing Values" in selected_steps:
            if fill_option == "Fill with Average":
                df = df.fillna(df.mean())
            elif fill_option == "Fill with Forward Fill":
                df = df.fillna(method='ffill')
        if "Remove Duplicates" in selected_steps:
            df = df.drop_duplicates()
    return df

# Function to visualize EDA
def eda_summary(df):
    st.write("### Exploratory Data Analysis")
    st.write("#### Summary Statistics:")
    st.write(df.describe())

    st.write("#### Missing Values:")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0] if not missing_values.empty else "No missing values.")
    
    st.write("#### Duplicates:")
    st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

# Function to apply models
def apply_model(df, model_type, target_column, x_variable):
    X = df[[x_variable]]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=200)
    elif model_type == 'SVM':
        model = SVC()
    elif model_type == 'Clustering':
        model = KMeans(n_clusters=3)
    else:
        st.error("Unknown model type selected.")
        return None, None

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model

# Main App
def app():
    st.title("Machine Learning Analysis Dashboard")
    
    # Home - Big Buttons Navigation
    st.write("## Navigate Through Stages of Data Analysis")
    st.write("### **1. Data Cleaning 2. Data Visualization 3. Data analysis, 4. Data Prediction.**")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    if col1.button("1. Data Cleaning"):
        st.session_state['nav'] = "Data Cleaning"
    if col2.button("2. Data Visualization"):
        st.session_state['nav'] = "Data Visualization"
    if col3.button("3. Data Analysis"):
        st.session_state['nav'] = "Data Analysis"
    if col4.button("4. Data Prediction"):
        st.session_state['nav'] = "Data Prediction"
    
    # Data Cleaning Tab
    if 'nav' in st.session_state and st.session_state['nav'] == "Data Cleaning":
        uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"], key="upload_csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            cleaning_steps = propose_cleaning_steps(df)
            selected_steps = st.radio("Select Cleaning Steps", cleaning_steps)
            fill_option = None
            if "Handle Missing Values" in selected_steps:
                fill_option = st.radio("How would you like to handle missing values?", ["Fill with Average", "Fill with Forward Fill"])
            if st.button("Clean Data"):
                clean_df = clean_data(df, selected_steps, fill_option)
                st.write("Cleaned Data:")
                st.dataframe(clean_df)
                st.session_state['clean_df'] = clean_df
        else:
            st.warning("Please upload a dataset to start cleaning.")

    # Data Visualization Tab
    if 'nav' in st.session_state and st.session_state['nav'] == "Data Visualization":
        if 'clean_df' in st.session_state:
            st.write("### Data Visualization")
            st.write("Select the type of visualization:")
            visualization_type = st.radio("Choose Visualization Type:", [
                "Histogram", "Correlation Heatmap", "Scatter Plot", "Box Plot", 
                "Pair Plot", "Line Plot", "Violin Plot", "Bar Plot", 
                "Area Plot", "Pie Chart", "3D Scatter Plot"
            ])
            
            # X and Y variable selection
            x_variable = st.selectbox("Select X (Independent) Variable:", st.session_state['clean_df'].columns)
            y_variable = st.selectbox("Select Y (Dependent) Variable:", st.session_state['clean_df'].columns)
            
            if visualization_type == "Histogram":
                selected_column = st.selectbox("Select Column for Histogram:", st.session_state['clean_df'].columns)
                fig, ax = plt.subplots()
                sns.histplot(st.session_state['clean_df'][selected_column], bins=30, kde=True, ax=ax)
                ax.set_title(f'Histogram of {selected_column}', fontsize=16)
                st.pyplot(fig)
                
            elif visualization_type == "Correlation Heatmap":
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(st.session_state['clean_df'].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

            elif visualization_type == "Scatter Plot":
                fig, ax = plt.subplots()
                sns.scatterplot(data=st.session_state['clean_df'], x=x_variable, y=y_variable, ax=ax)
                ax.set_title(f'Scatter Plot of {y_variable} vs {x_variable}', fontsize=16)
                st.pyplot(fig)

            elif visualization_type == "Box Plot":
                fig, ax = plt.subplots()
                sns.boxplot(data=st.session_state['clean_df'], x=x_variable, y=y_variable, ax=ax)
                ax.set_title(f'Box Plot of {y_variable} by {x_variable}', fontsize=16)
                st.pyplot(fig)

            elif visualization_type == "Pair Plot":
                if st.button("Show Pair Plot"):
                    pair_plot = sns.pairplot(st.session_state['clean_df'])
                    st.pyplot(pair_plot.fig)

            elif visualization_type == "Line Plot":
                fig, ax = plt.subplots()
                sns.lineplot(data=st.session_state['clean_df'], x=x_variable, y=y_variable, ax=ax)
                ax.set_title(f'Line Plot of {y_variable} vs {x_variable}', fontsize=16)
                st.pyplot(fig)

            elif visualization_type == "Violin Plot":
                fig, ax = plt.subplots()
                sns.violinplot(data=st.session_state['clean_df'], x=x_variable, y=y_variable, ax=ax)
                ax.set_title(f'Violin Plot of {y_variable} by {x_variable}', fontsize=16)
                st.pyplot(fig)

            elif visualization_type == "Bar Plot":
                fig, ax = plt.subplots()
                sns.barplot(data=st.session_state['clean_df'], x=x_variable, y=y_variable, ax=ax)
                ax.set_title(f'Bar Plot of {y_variable} by {x_variable}', fontsize=16)
                st.pyplot(fig)

            elif visualization_type == "Area Plot":
                fig, ax = plt.subplots()
                st.session_state['clean_df'].set_index(x_variable)[y_variable].plot(kind='area', ax=ax)
                ax.set_title(f'Area Plot of {y_variable} over {x_variable}', fontsize=16)
                st.pyplot(fig)

            elif visualization_type == "Pie Chart":
                fig, ax = plt.subplots()
                st.session_state['clean_df'][y_variable].value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
                ax.set_title(f'Pie Chart of {y_variable}', fontsize=16)
                st.pyplot(fig)

            elif visualization_type == "3D Scatter Plot":
                fig = px.scatter_3d(st.session_state['clean_df'], x=x_variable, y=y_variable, z=st.session_state['clean_df'].columns[0], title=f'3D Scatter Plot of {y_variable} vs {x_variable}')
                st.plotly_chart(fig)
        else:
            st.warning("Please clean the data first before visualizing.")

    # Data Analysis Tab
    if 'nav' in st.session_state and st.session_state['nav'] == "Data Analysis":
        if 'clean_df' in st.session_state:
            st.write("### Data Analysis")
            st.write("Performing analysis on cleaned data...")
            
            # Selecting target and independent variable
            target_column = st.selectbox("Select Target Variable (Y):", st.session_state['clean_df'].columns)
            x_variable = st.selectbox("Select Independent Variable (X):", st.session_state['clean_df'].columns)

            model_type = st.radio("Select Model Type:", ["Linear Regression", "Logistic Regression", "SVM", "Clustering"])
            
            if st.button("Apply Model"):
                predictions, model = apply_model(st.session_state['clean_df'], model_type, target_column, x_variable)
                
                if predictions is not None:
                    st.write("### Model Predictions:")
                    st.write(predictions)

                    # Visualizing the results
                    st.write("### Results Visualization:")
                    fig, ax = plt.subplots()
                    ax.scatter(st.session_state['clean_df'][x_variable], st.session_state['clean_df'][target_column], color='blue', label='Data')
                    
                    if model_type == 'Linear Regression':
                        ax.plot(st.session_state['clean_df'][x_variable], model.predict(st.session_state['clean_df'][[x_variable]]), color='red', label='Regression Line')
                    
                    ax.set_title(f'Predictions of {target_column} vs {x_variable}', fontsize=16)
                    ax.set_xlabel(x_variable)
                    ax.set_ylabel(target_column)
                    ax.legend()
                    st.pyplot(fig)

                    # Explain results
                    st.write("### Explanation of Results:")
                    st.write(f"The model predicts {target_column} based on {x_variable}. The red line represents the fitted model.")
        else:
            st.warning("Please clean the data first before analysis.")

    # Data Prediction Tab
    if 'nav' in st.session_state and st.session_state['nav'] == "Data Prediction":
        if 'clean_df' in st.session_state:
            st.write("### Model Prediction")
            target_column = st.selectbox("Select Target Variable:", st.session_state['clean_df'].columns)
            st.write("### Select Features for Prediction")
            features = st.multiselect("Choose Features:", st.session_state['clean_df'].columns.tolist())
            
            if st.button("Make Prediction"):
                if len(features) == 0:
                    st.error("Please select at least one feature.")
                else:
                    predictions = apply_model(st.session_state['clean_df'], 'linear_regression', target_column, features[0])  # Using the first feature for demo
                    st.write("Predictions based on selected features:", predictions)
        else:
            st.warning("Please clean the data first before making predictions.")

# Run the app
if __name__ == "__main__":
    app()
