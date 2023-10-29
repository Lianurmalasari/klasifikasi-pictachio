import pickle
import streamlit as st
import numpy as np

model = pickle.load(open('klasifikasi-pistachio.sav', 'rb'))
scaler = pickle.load(open('scaler.sav','rb'))

st.title('Prediksi Jenis Pistachio')

#membagi kolom
col1, col2 = st.columns(2)

with col1 :
    AREA = st.number_input('Luas pistachio')
    PERIMETER = st.number_input('Keliling pistachio')
    MAJOR_AXIS = st.number_input('Sumbu utama pistachio')
    MINOR_AXIS = st.number_input('Sumbu minor pistachio')
    ECCENTRICITY = st.number_input('Eksentrisitas pistachio')
    EQDIASQ = st.number_input('Luas sama dengan diameter persegi pistachio')

with col2 :
    SOLIDITY = st.number_input('Kepadatan pistachio')
    CONVEX_AREA = st.number_input('Luas cembung pistachio')
    EXTENT = st.number_input('Tingkat perluasan pistachio')
    ASPECT_RATIO = st.number_input('Rasio aspek pistachio')
    ROUNDNESS = st.number_input('Kebulatan pistachio')
    COMPACTNESS = st.number_input('Kekompakan pistachio')

prediction = ''
input_data = (AREA,PERIMETER,MAJOR_AXIS,MINOR_AXIS,ECCENTRICITY,EQDIASQ,SOLIDITY,
              CONVEX_AREA,EXTENT,ASPECT_RATIO,ROUNDNESS,COMPACTNESS)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

if st.button('Proses'):
    prediction = model.predict(std_data)
    if(prediction[0] == 1):
        prediction = 'Jenis Kirmizi Pistachio'
    else:
        prediction = 'Jenis Siit Pistachio'
    st.success(prediction)
