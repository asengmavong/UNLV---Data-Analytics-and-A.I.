#Streamlit application
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

#Page Config
st.set_page_config(layout="wide")

#Cleaning function to handle yes/no columns and unwanted columns

def clean_df(df, yes_no_cols, cols_to_drop=None):
    
    """
    Cleans data file to be used for machine learning by mapping yes/no to 1 and 0, 
    creating dummy variables, and converting time to datetime.
    """
    
    # Create a copy of the dataframe
    cleaned_df = df.copy()

    # Strip whitespace from every string cell in the entire DataFrame
    # Safely ignores integers, floats, and NaNs
    cleaned_df = cleaned_df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Remove unwanted columns
    if cols_to_drop:
        cleaned_df = cleaned_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Convert Yes and No columns to 1 and 0
    if yes_no_cols:
        # Standardize strings to lowercase and strip whitespaces
        for col in yes_no_cols:
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
            # Map values (unknown/other values will become NaN)
            cleaned_df[col] = cleaned_df[col].replace({'yes': 1, 'y': 1, 'no': 0, 'n': 0}).astype(int)
        
    return cleaned_df

#Load HR Attrition Model
with open('Attrition_model.pkl', 'rb') as file:
    model = pickle.load(file)

#HR data input sidebar
def user_input():

    st.sidebar.header("HR Data Input Features")
    
    EnvironmentSatisfaction = st.sidebar.number_input('EnvironmentSatisfaction', min_value=1, max_value=5, step=1, value=4)

    JobSatisfaction = st.sidebar.number_input('JobSatisfaction', min_value=1, max_value=5, step=1, value=4)

    RelationshipSatisfaction = st.sidebar.number_input('RelationshipSatisfaction', min_value=1, max_value=5, step=1, value=4)

    TrainingOpportunitiesWithinYear = st.sidebar.number_input('TrainingOpportunitiesWithinYear', min_value=1, max_value=5, step=1, value=4)

    TrainingOpportunitiesTaken = st.sidebar.number_input('TrainingOpportunitiesTaken', min_value=1, max_value=5, step=1, value=4)

    WorkLifeBalance = st.sidebar.number_input('WorkLifeBalance', min_value=1, max_value=5, step=1, value=4)

    SelfRating = st.sidebar.number_input('SelfRating', min_value=1, max_value=5, step=1, value=4)

    ManagerRating = st.sidebar.number_input('ManagerRating', min_value=1, max_value=5, step=1, value=4)
        
    Gender = st.sidebar.selectbox('Gender',['Female', 'Male', 'Non-Binary','Prefer Not To Say'])
    
    Age = st.sidebar.number_input('Age', min_value=1, max_value=65, step=1, value=30)
    
    BusinessTravel = st.sidebar.selectbox('BusinessTravel',['Some Travel', 'Frequent Traveller', 'Non-No Travel'])
    
    Department = st.sidebar.selectbox('Department',['Technology', 'Sales', 'Human Resources'])
    
    DistanceFromHome_KM = st.sidebar.number_input('DistanceFromHome_KM', min_value=0, max_value=100, step=1, value=30)
    
    State = st.sidebar.selectbox('State',['CA', 'NY', 'IL'])
    
    Ethnicity = st.sidebar.selectbox('Ethnicity',['White', 'Black or African American', 'Asian or Asian American',
                                                 'American Indian or Alaska Native', 'Native Hawaiian',
                                                 'Mixed or multiple ethnic groups', 'Other'])
    
    Education = st.sidebar.number_input('Education', min_value=1, max_value=5, step=1, value=3)
    
    EducationField = st.sidebar.selectbox('EducationField',['Computer Science', 'Marketing', 'Information Systems',
                                                 'Business Studies', 'Economics','Human Resources',
                                                 'Technical Degree', 'Other'])
    
    JobRole = st.sidebar.selectbox('JobRole',['Sales Executive','Software Engineer','Data Scientist',
                                   'Machine Learning Engineer','Senior Software Engineer',
                                  'Engineering Manager','Sales Representative',
                                   'Analytics Manager','Manager','HR Executive','Recruiter',
                                   'HR Business Partner','HR Manager'])
    
    MaritalStatus = st.sidebar.selectbox('MaritalStatus',['Single','Married','Divorced'])
    
    Salary = st.sidebar.number_input('Salary', min_value=0, max_value=1000000, step=1, value=100000)
    
    StockOptionLevel = st.sidebar.selectbox('StockOptionLevel',['Yes','No'])
    
    OverTime = st.sidebar.selectbox('OverTime',['Yes','No'])
    
    YearsAtCompany = st.sidebar.number_input('YearsAtCompany', min_value=0, max_value=65, step=1, value=30)
    
    YearsInMostRecentRole = st.sidebar.number_input('YearsInMostRecentRole', min_value=0, max_value=65, step=1, value=30)
    
    YearsSinceLastPromotion = st.sidebar.number_input('YearsSinceLastPromotion', min_value=0, max_value=65, step=1, value=30)
    
    YearsWithCurrManager = st.sidebar.number_input('YearsWithCurrManager', min_value=0, max_value=65, step=1, value=30)

    user_data = {
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'JobSatisfaction': JobSatisfaction,
        'RelationshipSatisfaction': RelationshipSatisfaction,
        'TrainingOpportunitiesWithinYear': TrainingOpportunitiesWithinYear,
        'TrainingOpportunitiesTaken': TrainingOpportunitiesTaken,
        'WorkLifeBalance': WorkLifeBalance,
        'SelfRating': SelfRating,
        'ManagerRating': ManagerRating,
        'Gender': Gender,
        'Age': Age,
        'BusinessTravel': BusinessTravel,
        'Department': Department,
        'DistanceFromHome_KM': DistanceFromHome_KM,
        'State': State,
        'Ethnicity': Ethnicity,
        'Education': Education,
        'EducationField': EducationField,
        'JobRole': JobRole,
        'MaritalStatus': MaritalStatus,
        'Salary': Salary,
        'StockOptionLevel': StockOptionLevel,
        'OverTime': OverTime,
        'YearsAtCompany': YearsAtCompany,
        'YearsInMostRecentRole': YearsInMostRecentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager
    }
    features = pd.DataFrame([user_data])
    return features

# Centered title
st.markdown("<h1 style='text-align: center;'>Attrition Prediction App</h1>", unsafe_allow_html=True)

st.header("Predict Employee Attrition")

# Get inputs from sidebar
user_data = user_input()

# Predict button
if st.button("Predict"):
    prediction = model.predict(user_data)

    # Get probabilities for each class [Probability_of_0, Probability_of_1]
    probabilities = model.predict_proba(user_data)[0]
    
    # Extract the number from the array (usually index 0)
    pred_value = prediction[0]
    
    # Map the 0 or 1 value to Yes or No
    result = "Yes" if pred_value == 1 else "No"
    
    # probabilities[1] is the chance of leaving; probabilities[0] is the chance of staying
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    # Format as a percentage
    confidence_pct = f"{confidence * 100:.1f}%"
    
    # Display the results in Streamlit
    st.subheader("Will employee leave?")
    
    # Option A: Clean text layout
    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence:** {confidence_pct}")

# streamlit run HR_Attrition_predict.py
