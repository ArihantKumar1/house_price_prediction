import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
data = pd.read_csv("final_preprocessed_dataset.csv")

# Extract feature names from the dataset
feature_names = data.drop(columns=['price']).columns  # Exclude 'price' from the feature names

# Load the trained models
with open('models.pkl', 'rb') as file:
    models = pickle.load(file)

# Streamlit app setup
st.set_page_config(page_title="House Price Prediction", layout="wide", initial_sidebar_state="expanded")

# App title and header
st.title("🏠 House Price Prediction")
st.markdown("""
This application predicts house prices based on user inputs using different machine learning models.
Use the sidebar to input the features and select a model for prediction.
""")

# Sidebar for input features
st.sidebar.header("🔧 Input Features")

# Input fields for numerical features
beds = st.sidebar.number_input("Number of Bedrooms (Beds)", min_value=0, step=1, value=3)
baths = st.sidebar.number_input("Number of Bathrooms (Baths)", min_value=0, step=1, value=2)
size = st.sidebar.number_input("Size (Square Feet)", min_value=0, step=1, value=1500)
price_per_sqft = st.sidebar.number_input("Price Per Square Foot", min_value=0.0, step=1.0, value=200.0)
zip_code = st.sidebar.number_input("ZIP Code", min_value=0, step=1, value=12345)

# Collect categorical inputs dynamically based on preprocessed data
categorical_features = [col for col in feature_names if col.startswith('category_')]
category_inputs = {}

for feature in categorical_features:
    category = feature.replace('category_', '')
    category_inputs[feature] = st.sidebar.selectbox(f"Category: {category}", options=[0, 1])

# Combine inputs into a DataFrame
input_data = pd.DataFrame({
    "beds": [beds],
    "baths": [baths],
    "size": [size],
    "price_per_sqft": [price_per_sqft],
    "zip_code": [zip_code],
    **{key: [value] for key, value in category_inputs.items()}
})

# Align input_data columns with the expected feature names
for feature in feature_names:
    if feature not in input_data:
        input_data[feature] = 0  # Add missing features with default value 0

# Ensure the columns are in the same order as in the original dataset
input_data = input_data[feature_names]

# StandardScaler for scaling the input data
scaler = StandardScaler()
scaler.fit(data.drop(columns=['price']))  # Fit the scaler on the dataset used for training

# Sidebar for model selection
model_name = st.sidebar.selectbox("⚙️ Select Model", options=list(models.keys()))

# Predict the price
if st.sidebar.button("🔍 Predict"):
    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Retrieve the selected model
    model = models[model_name]

    # Make predictions
    predicted_price = model.predict(input_scaled)[0]

    # Display the prediction result
    st.subheader("📊 Prediction Result")
    st.write(f"**Selected Model:** {model_name}")
    st.write(f"**Predicted House Price:** ${predicted_price:,.2f}")

    # Visualization Section
    st.subheader("📈 Visualizations")

    # Feature Importance Plot for Decision Tree and Random Forest
    if model_name in ["Decision Tree", "Random Forest"]:
        if hasattr(model, 'feature_importances_'):
            st.markdown("### Feature Importance")
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)

            # Plotting
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
            plt.title(f"Feature Importance: {model_name}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            st.pyplot(plt)
    else:
        st.markdown("### No visualizations available for this model.")
