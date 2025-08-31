import streamlit as st
import pandas as pd
import joblib
import sklearn.compose._column_transformer as ct

# Dummy class to satisfy the pickle
class _RemainderColsList(list):
    pass

ct._RemainderColsList = _RemainderColsList

# Load the trained model
with open("logistic_regression_model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("🚢 Titanic Survival Prediction")

# -------------------------
# Create a form for inputs
# -------------------------
with st.form(key='titanic_form'):
    pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
    parch = st.number_input("Number of Parents/Children Aboard", min_value=0, value=0)
    fare = st.number_input("Fare Paid in £ (check notes)", min_value=0.0, value=7.25)
# Display typical ticket prices for context
    sex = st.selectbox("Sex", ["male", "female"])
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
    submit_button = st.form_submit_button(label='Predict Survival')

    st.markdown("""
    **Typical Titanic Ticket Prices by Class:**
    - **First Class:** Approx. £30 to £870 (roughly $150–$4,350)
    - **Second Class:** Approx. £12 to £60 ($60–$300)
    - **Third Class (Steerage):** Approx. £3 to £8 ($15–$40)
    """)
# -------------------------
# Prepare data and predict
# -------------------------
if submit_button:
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Sex": sex,
        "Embarked": embarked
    }])

    # Feature engineering
    input_data['FamilySize'] = input_data['SibSp'] + input_data['Parch'] + 1

    def get_age_group(age):
        if age < 12:
            return 'Child'
        elif age < 20:
            return 'Teen'
        elif age < 60:
            return 'Adult'
        else:
            return 'Senior'

    input_data['AgeGroup'] = input_data['Age'].apply(get_age_group)

    # Add placeholder Title column
    input_data['Title'] = 'Mr'  # placeholder to satisfy pipeline

    # Prediction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]

        result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
        st.subheader(result)
        st.write(f"Survival Probability: {prediction_proba:.2%}")
    except Exception as e:
        st.error(f"Error: {e}")
