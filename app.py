import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------
# Load Models
# -------------------------

car_model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pickle.load(open('Cleaned_Car_data.pkl', 'rb'))

pipe = pickle.load(open('pipe.pkl', 'rb'))
laptop_df = pickle.load(open('df.pkl', 'rb'))

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("AI Price Predictor Hub")

page = st.sidebar.selectbox(
    "Choose Predictor",
    ["Car Price Predictor", "Laptop Price Predictor"]
)

# -------------------------
# CAR PREDICTOR
# -------------------------

if page == "Car Price Predictor":

    st.title("🚗 Car Price Predictor")

    name = st.selectbox('Car Name', car['name'].unique())
    company = st.selectbox('Company', car['company'].unique())
    year = st.selectbox('Year', sorted(car['year'].unique(), reverse=True))
    fuel_type = st.selectbox('Fuel Type', car['fuel_type'].unique())
    kms_driven = st.number_input('Kms Driven')

    if st.button('Predict Car Price'):

        prediction = car_model.predict(
            pd.DataFrame(
                [[name, company, year, kms_driven, fuel_type]],
                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
            )
        )

        st.success(f"Predicted Price: ₹ {int(prediction[0]):,}")

# -------------------------
# LAPTOP PREDICTOR
# -------------------------

elif page == "Laptop Price Predictor":

    st.title("💻 Laptop Price Predictor")

    company = st.selectbox('Company', laptop_df['Company'].unique())
    typename = st.selectbox('Type', laptop_df['TypeName'].unique())

    inches = st.number_input('Screen Size (Inches)', value=15.6)

    ram = st.selectbox('RAM (GB)', sorted(laptop_df['Ram'].unique()))

    weight = st.number_input('Weight', value=2.0)

    touchscreen = st.selectbox(
        'Touchscreen',
        ['No', 'Yes']
    )

    ips = st.selectbox(
        'IPS Display',
        ['No', 'Yes']
    )

    ppi = st.number_input('PPI', value=141.0)

    cpu = st.selectbox(
        'CPU Brand',
        laptop_df['Cpu brand'].unique()
    )

    hdd = st.selectbox(
        'HDD (GB)',
        sorted(laptop_df['HDD'].unique())
    )

    ssd = st.selectbox(
        'SSD (GB)',
        sorted(laptop_df['SSD'].unique())
    )

    gpu = st.selectbox(
        'GPU Brand',
        laptop_df['Gpu brand'].unique()
    )

    os = st.selectbox(
        'Operating System',
        laptop_df['os'].unique()
    )

    if st.button("Predict Laptop Price"):

        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0

        query = pd.DataFrame(
            [[
                company,
                typename,
                inches,
                ram,
                weight,
                touchscreen_val,
                ips_val,
                ppi,
                cpu,
                hdd,
                ssd,
                gpu,
                os
            ]],
            columns=[
                'Company',
                'TypeName',
                'Inches',
                'Ram',
                'Weight',
                'Touchscreen',
                'IPS',
                'ppi',
                'Cpu brand',
                'HDD',
                'SSD',
                'Gpu brand',
                'os'
            ]
        )

        prediction = pipe.predict(query)

        st.success(
            f"Predicted Laptop Price: ₹ {int(np.exp(prediction[0])):,}"
        )

