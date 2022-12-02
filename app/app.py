import streamlit as st
import requests

st.sidebar.title('Welcome on Cardio bot!🩺 \n  ')
st.sidebar.subheader('Your autonomous heartbeat checkup')

st.sidebar.write('Dear Patient,')
st.sidebar.write("On this online platform \
                you will be able to run preliminary health checks \
                before going to your cardiologist.")
st.sidebar.markdown('---')
st.sidebar.title('How is it working ?')
st.sidebar.write(" 1) Cardio bot will first convert your heart recording into a spectogram. \
                This way, you will be able to see a visual representation of your heartbeat")
st.sidebar.write(" 2) In a second step, Cardio bot will \
                detect if your heart has a murmur or not")
phonocardiogram = st.sidebar.file_uploader(label="Upload your phonocardiogram below :",
                                           type=[".wav", ".wave", ".flac", ".mp3", ".ogg"],)

with st.sidebar.form(key='How is my heart looking ?'):
    submit_button = st.form_submit_button(label='How is my heart looking 💓? ')

if submit_button:
    url = 'https://circordck-pz3kchuqsq-ew.a.run.app/show'

    if url == 'https://circordck-pz3kchuqsq-ew.a.run.app/show':

        parameters = {}

        response = requests.get(url,params = parameters, timeout=10).json()

        st.image(f'you ride will cost: {response} $ ')
