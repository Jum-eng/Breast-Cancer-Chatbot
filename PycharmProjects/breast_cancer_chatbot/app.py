import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

# Define the feature columns
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Streamlit App
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="wide",
    page_icon="🩺"  # You can replace this with any emoji or URL to an image
)

# Add a banner image
st.image("img.png")

st.title("Breast Cancer Prediction Chatbot")
st.markdown("""
### Early detection saves lives. Provide the details below to predict the likelihood of breast cancer.
This application uses a trained machine learning model to predict whether a tumor is **malignant** or **benign** based on clinical data.
""")

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info("""
This tool is designed to assist medical professionals and individuals in predicting breast cancer risk. 

For accurate predictions, ensure all inputs are based on medical tests.
""")

st.sidebar.title("Contact Us")
st.sidebar.write("""
For feedback or inquiries:

Email: jumahmusah87@gmail.com
 
Phone: +233 24 528 7120
""")

# Input form with sections
with st.form("breast_cancer_form"):
    st.header("Input Clinical Features")
    user_inputs = {}

    # Group inputs into collapsible sections
    with st.expander("Mean Features"):
        for feature in features[:10]:
            user_inputs[feature] = st.number_input(f"Enter {feature.replace('_', ' ').capitalize()}", min_value=0.0,
                                                   step=0.01)

    with st.expander("SE Features"):
        for feature in features[10:20]:
            user_inputs[feature] = st.number_input(f"Enter {feature.replace('_', ' ').capitalize()}", min_value=0.0,
                                                   step=0.01)

    with st.expander("Worst Features"):
        for feature in features[20:]:
            user_inputs[feature] = st.number_input(f"Enter {feature.replace('_', ' ').capitalize()}", min_value=0.0,
                                                   step=0.01)

    submitted = st.form_submit_button("Predict")

    # Prediction logic in app.py
    if submitted:
        # Convert user input into a DataFrame
        input_data = pd.DataFrame([user_inputs])

        try:
            # Make prediction
            prediction_proba = model.predict_proba(input_data)  # Get probability estimates

            # Set a custom threshold
            threshold = 0.6  # Adjust as needed

            # Make the prediction based on the threshold
            if prediction_proba[0][1] > threshold:
                st.error(
                    f"The model predicts the tumor is **malignant** with a probability of {prediction_proba[0][1]:.2f}. Please consult a doctor."
                )
            else:
                st.success(
                    f"The model predicts the tumor is **benign** with a probability of {prediction_proba[0][0]:.2f}."
                )

            # Display probabilities as a chart
            st.subheader("Prediction Probability")
            st.bar_chart({"Malignant": prediction_proba[0][1], "Benign": prediction_proba[0][0]})

        except ValueError as e:
            st.error(f"Error: {e}")
