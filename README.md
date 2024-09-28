# AIAutomatedDashboard
AI Powered Dashboard - Do All Data Analysis Easily (no code required)

Here’s a detailed README for this Streamlit dashboard that outlines the functionalities, navigation instructions, and descriptions of buttons and pages.

Overview
This dashboard is designed for comprehensive data cleaning, visualization, and analysis, tailored for users, especially beginners in data analysis. It incorporates an intuitive user interface, dynamic visualizations, and interactive analysis features to enhance user experience and understanding.

Key Functionalities
1. Data Cleaning
Upload a Dataset: Users can upload a CSV file for analysis.
Cleaning Steps: Users can select specific cleaning steps:
Handle Missing Values: Options to fill missing values either by the average or forward fill method.
Remove Duplicates: Identifies and removes duplicate rows.
Propose Cleaning Steps: The dashboard automatically checks for missing values, duplicates, and inconsistent formatting, suggesting necessary cleaning steps.
2. Data Visualization
Visualization Types: Users can choose from various types of visualizations, including:
Histogram
Correlation Heatmap
Scatter Plot
Box Plot
Pair Plot
Line Plot
Violin Plot
Bar Plot
Area Plot
Pie Chart
3D Scatter Plot
Dynamic Variable Selection: Users can select independent (X) and dependent (Y) variables, with hues option for additional data representation.
3D and Time Series Support: Users can visualize data in 3D and time series formats.
3. Data Analysis
Predictive Analysis: Users can apply different models (Linear Regression, SVM, Clustering) to the cleaned dataset.
Model Visualizations: Users can visualize results from the applied models, including displaying regression lines and SVM hyperplanes.
4. User Interface Enhancements
Dark Theme: The dashboard features a dark metallic background for a modern look.
Big Buttons: Navigation buttons are large and easily accessible.
Sequential Navigation: Users are guided to navigate through the pages in a specific order.
Navigation Instructions
Home Page
Purpose: This is the landing page of the dashboard.
Navigation Buttons:
Data Cleaning: Click to enter the data cleaning section.
Data Visualization: Click to go to the data visualization section.
Data Analysis: Click to access the data analysis section.
Data Prediction: Click to proceed to the prediction section (to be implemented).
Data Cleaning Page
Upload Dataset:
File Uploader: Click to upload your CSV file.
Select Cleaning Steps:
Radio Buttons: Choose steps to clean the data (e.g., handle missing values, remove duplicates).
Fill Options:
Radio Buttons: If you select to handle missing values, choose how to fill them.
Clean Data Button:
Action: Click this button to execute the selected cleaning steps and display the cleaned data.
Data Visualization Page
Visualization Type:
Radio Buttons: Select the type of visualization you want to create.
Variable Selection:
Dropdowns: Choose independent and dependent variables from the cleaned dataset.
Hues Selection: Choose a hues variable after the main selections.
Visualization Output:
The chosen visualization will be displayed below based on the selections made.
Data Analysis Page
Target Variable:
Dropdown: Select the target variable for prediction.
Independent Variable:
Dropdown: Choose an independent variable for the model.
Model Selection:
Radio Buttons: Select the type of model to apply (Linear Regression, SVM, Clustering).
Apply Model Button:
Action: Click to run the selected model and display predictions and visualizations.
Data Prediction Page
To be implemented: Future functionalities may include additional predictive features and analyses.
Business Rules
Data Handling: All data cleaning operations must be applied sequentially; ensure the dataset is clean before visualization and analysis.
User Guidance: The dashboard provides warnings and notifications to guide users through necessary actions and inform them about the state of the data.
Visualizations: Ensure visualizations are selected based on the variable types; some visualizations require specific data formats or counts.
Future Enhancements
Implementation of more advanced machine learning models.
Integration of user-uploaded files for model training and testing.
Enhanced interactivity and responsiveness in visualizations.
Contact Information
For any questions or feedback regarding the dashboard, please contact [Your Name] at [Your Email].
