import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import numpy as np

# Create a title for the app
st.title("Standalone Machine Learning Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Data Summary:")
    st.write(data.describe())
    
    # Feature selection
    x_col = st.selectbox("Select X (Feature) for Analysis", data.columns)
    y_col = st.selectbox("Select Y (Target) for Analysis", data.columns)

    # Check if the target is categorical for classification tasks
    if data[y_col].nunique() <= 10:
        is_classification = True
    else:
        is_classification = False

    # Train/Test split
    test_size = st.slider("Train/Test Split", min_value=0.1, max_value=0.9, value=0.2)

    # Algorithm selection
    algorithms = st.multiselect("Choose Machine Learning Algorithm(s)", 
                                 ["Linear Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Logistic Regression"])

    # Button to train the selected model
    if st.button("Train Model"):
        # Split the data
        X = data[[x_col]]
        y = data[y_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize a dictionary to store models
        model_dict = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "SVM": SVR(),
            "KNN": KNeighborsRegressor(),
            "Logistic Regression": LogisticRegression(),
            "Decision Tree (Classifier)": DecisionTreeClassifier(),
            "Random Forest (Classifier)": RandomForestClassifier(),
            "SVM (Classifier)": SVC(),
            "KNN (Classifier)": KNeighborsClassifier()
        }

        # Train selected models and display metrics
        for algo in algorithms:
            if algo in model_dict:
                model = model_dict[algo]
                try:
                    model.fit(X_train, y_train)
                except ValueError as e:
                    st.error(f"Error while fitting the model: {e}")
                    continue

                if is_classification and "Logistic Regression" in algo:
                    predictions = model.predict(X_test)

                    # Display confusion matrix and classification metrics
                    st.write(f"**Confusion Matrix for {algo}:**")
                    cm = confusion_matrix(y_test, predictions)
                    st.write(cm)

                    report = classification_report(y_test, predictions, output_dict=True)
                    st.write("**Classification Report:**")
                    st.write(f"**Precision:** {report['weighted avg']['precision']:.2f}")
                    st.write(f"**Recall:** {report['weighted avg']['recall']:.2f}")
                    st.write(f"**F1 Score:** {report['weighted avg']['f1-score']:.2f}")

                elif not is_classification:
                    predictions = model.predict(X_test)
                    st.write(f"**Predictions for {algo}:**", predictions)

                    # Display error metrics with explanations
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)

                    st.write(f"**Mean Squared Error (MSE) for {algo}:** {mse:.2f} (Lower is better)")
                    st.write("MSE measures the average squared difference between actual and predicted values. Acceptable values depend on the context; smaller values indicate better model performance.")
                    st.write(f"**R^2 Score for {algo}:** {r2:.2f} (Closer to 1 is better)")
                    st.write("R^2 Score indicates the proportion of variance in the target variable that is predictable from the features. A value closer to 1 indicates a better fit.")

                    # Visualization of Results
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=y_test.values.flatten(), y=predictions, ax=ax)
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title(f"Actual vs Predicted for {algo}")
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Add a reference line
                    st.pyplot(fig)

                    # Residuals plot
                    residuals = y_test.values.flatten() - predictions
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=predictions, y=residuals, ax=ax)
                    ax.hlines(0, predictions.min(), predictions.max(), colors='red', linestyles='--')
                    ax.set_xlabel("Predicted Values")
                    ax.set_ylabel("Residuals")
                    ax.set_title(f"Residuals for {algo}")
                    st.pyplot(fig)

else:
    st.info("Please upload a CSV or Excel file to start.")
