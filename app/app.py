import streamlit as st
import requests
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from PIL import Image

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
                                           type='.wav')

with st.sidebar.form(key='How is my heart looking ?'):
    submit_button = st.form_submit_button(label='How is my heart looking 💓? ')

if submit_button:

    st.audio(phonocardiogram, format="audio.wav", start_time=0)

    phonocardiogram, fs = librosa.load(phonocardiogram, sr=None)
    phonocardiogram = np.array(phonocardiogram)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(phonocardiogram)), ref=np.max)

    librosa.display.specshow(D, sr=fs, y_axis='linear', x_axis='time')
    st.pyplot()
