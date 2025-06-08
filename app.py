
import streamlit as st
import joblib
import numpy as np

# Load your saved model
model = joblib.load('random_forest_model.pkl')

st.title("E-commerce Delivery Time Prediction")

# Numeric inputs
agent_age = st.number_input("Agent Age", min_value=18, max_value=80, step=1)
agent_rating = st.number_input("Agent Rating (1-5)", min_value=1.0, max_value=5.0, step=0.1)
store_lat = st.number_input("Store Latitude", format="%.6f")
store_lon = st.number_input("Store Longitude", format="%.6f")
drop_lat = st.number_input("Drop Latitude", format="%.6f")
drop_lon = st.number_input("Drop Longitude", format="%.6f")
distance_km = st.number_input("Distance (km)", min_value=0.0, step=0.1)
order_hour = st.number_input("Order Hour (0-23)", min_value=0, max_value=23, step=1)
order_day = st.number_input("Order Day of Week (0=Monday)", min_value=0, max_value=6, step=1)
pickup_delay = st.number_input("Pickup Delay (minutes)", min_value=0, step=1)

# Weather options
weather_options = ['Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy']
weather = st.selectbox("Weather Condition", weather_options)

weather_fog = 1 if weather == 'Fog' else 0
weather_sandstorms = 1 if weather == 'Sandstorms' else 0
weather_stormy = 1 if weather == 'Stormy' else 0
weather_sunny = 1 if weather == 'Sunny' else 0
weather_windy = 1 if weather == 'Windy' else 0

# Traffic options
traffic_options = ['Jam', 'Low', 'Medium']
traffic = st.selectbox("Traffic Condition", traffic_options)

traffic_jam = 1 if traffic == 'Jam' else 0
traffic_low = 1 if traffic == 'Low' else 0
traffic_medium = 1 if traffic == 'Medium' else 0

# Vehicle options
vehicle_options = ['scooter', 'van']
vehicle = st.selectbox("Vehicle Type", vehicle_options)

vehicle_scooter = 1 if vehicle == 'scooter' else 0
vehicle_van = 1 if vehicle == 'van' else 0

# Area options
area_options = ['Other', 'Semi-Urban', 'Urban']
area = st.selectbox("Delivery Area", area_options)

area_other = 1 if area == 'Other' else 0
area_semiurban = 1 if area == 'Semi-Urban' else 0
area_urban = 1 if area == 'Urban' else 0

# Category options
category_options = ['Books', 'Clothing', 'Cosmetics', 'Electronics', 'Grocery', 'Home', 'Jewelry', 'Kitchen', 'Outdoors',
                    'Pet Supplies', 'Shoes', 'Skincare', 'Snacks', 'Sports', 'Toys']
category = st.selectbox("Product Category", category_options)

category_books = 1 if category == 'Books' else 0
category_clothing = 1 if category == 'Clothing' else 0
category_cosmetics = 1 if category == 'Cosmetics' else 0
category_electronics = 1 if category == 'Electronics' else 0
category_grocery = 1 if category == 'Grocery' else 0
category_home = 1 if category == 'Home' else 0
category_jewelry = 1 if category == 'Jewelry' else 0
category_kitchen = 1 if category == 'Kitchen' else 0
category_outdoors = 1 if category == 'Outdoors' else 0
category_pet_supplies = 1 if category == 'Pet Supplies' else 0
category_shoes = 1 if category == 'Shoes' else 0
category_skincare = 1 if category == 'Skincare' else 0
category_snacks = 1 if category == 'Snacks' else 0
category_sports = 1 if category == 'Sports' else 0
category_toys = 1 if category == 'Toys' else 0

features = np.array([[
    agent_age,
    agent_rating,
    store_lat,
    store_lon,
    drop_lat,
    drop_lon,
    distance_km,
    order_hour,
    order_day,
    pickup_delay,
    weather_fog,
    weather_sandstorms,
    weather_stormy,
    weather_sunny,
    weather_windy,
    traffic_jam,
    traffic_low,
    traffic_medium,
    vehicle_scooter,
    vehicle_van,
    area_other,
    area_semiurban,
    area_urban,
    category_books,
    category_clothing,
    category_cosmetics,
    category_electronics,
    category_grocery,
    category_home,
    category_jewelry,
    category_kitchen,
    category_outdoors,
    category_pet_supplies,
    category_shoes,
    category_skincare,
    category_snacks,
    category_sports,
    category_toys
]])

if st.button("Predict Delivery Time"):
    prediction = model.predict(features)
    st.success(f"Predicted Delivery Time: {prediction[0]:.2f} hours")
