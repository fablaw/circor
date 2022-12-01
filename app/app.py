import streamlit as st
import requests

# One heart

phonocardiogram = st.file_uploader('Share your phonocardiogram with us')

with st.form(key='Get my fare!'):
    submit_button = st.form_submit_button(label='Get my fare price')

if submit_button:
    url = 'https://taxifare.lewagon.ai/predict'

    if url == 'https://taxifare.lewagon.ai/predict':

        parameters = {'audios': f'{ride_date} {ride_time}',
                'pickup_longitude' : pickup_longitude,
                'pickup_latitude' : pickup_latitude,
                'dropoff_longitude' : dropoff_longitude,
                'dropoff_latitude' : dropoff_latitude,
                'passenger_count' : passenger_count}

        response = requests.get(url,params = parameters, timeout=10).json()

        st.markdown(f'you ride will cost: {response} $ ')
