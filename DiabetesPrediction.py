import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def main():
    st.title("ML Model Deployment with Streamlit")
    st.write("This is a simple web app to predict outcomes based on your trained ML model.")


    feature1 = st.number_input("Pregnancies", min_value=0.0, step=1.0)
    feature2 = st.number_input("Glucose", min_value=0.0, step=1.0)
    feature3 = st.number_input("Blood Pressure", min_value=0.0, step=1.0)
    feature4 = st.number_input("Skin Thickness", min_value=0.0, step=1.0)
    feature5 = st.number_input("Insulin", min_value=0.0, step=1.0)
    feature6 = st.number_input("BMI", min_value=0.0, step=0.1)
    feature7 = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.001, format="%.3f")
    feature8 = st.number_input("Age", min_value=0, step=1)

   
    if st.button("Predict"):
       
        input_data = np.asarray([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8])

        input_data_reshaped = input_data.reshape(1, -1)
        std_data = scaler.transform(input_data_reshaped)

        prediction = model.predict(std_data)

        if prediction[0] ==1:
            st.warning("The given person is diabetic")
        else:
            st.success("The given person is not diabetic")

if __name__ == '__main__':
    main()
