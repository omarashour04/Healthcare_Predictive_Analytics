import streamlit as st
import joblib
import numpy as np

# Load encoders
label_encoder = joblib.load("saved_models/label_encoder.pkl")
mlb = joblib.load("saved_models/symptom_encoder.pkl")

# Load all models
models = {
    "Logistic Regression": joblib.load("saved_models/logistic_regression.pkl"),
    "SVM (Linear Kernel)": joblib.load("saved_models/svm_linear.pkl"),
    "Naive Bayes": joblib.load("saved_models/naive_bayes.pkl"),
    "Decision Tree": joblib.load("saved_models/decision_tree.pkl"),
    "XGBoost": joblib.load("saved_models/xgboost.pkl")
}

model_accuracies = {
    "Logistic Regression": 0.9979,
    "SVM (Linear Kernel)": 0.9979,
    "Naive Bayes": 0.9976,
    "Decision Tree": 0.9920,
    "XGBoost": 0.9979,
    "Random Forest": 0.9980
}


# Streamlit UI
st.title("🩺 Disease Prediction from Symptoms")
st.markdown("Select your symptoms and a model to predict the disease.")

# Select model
model_choice = st.selectbox("Choose a model:", list(models.keys()))
# Select model
# Display the selected model's accuracy
st.info(f"✅ Accuracy of **{model_choice}**: **{model_accuracies[model_choice]*100:.2f}%**")

# Select symptoms
all_symptoms = sorted(mlb.classes_)
selected_symptoms = st.multiselect("Choose symptoms:", all_symptoms)

# Predict
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        model = models[model_choice]
        input_symptoms = [s.strip().lower() for s in selected_symptoms]
        encoded_input = mlb.transform([input_symptoms])
        prediction = model.predict(encoded_input)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]

        st.success(f"✅ Predicted Disease: **{predicted_disease}** using {model_choice}")
