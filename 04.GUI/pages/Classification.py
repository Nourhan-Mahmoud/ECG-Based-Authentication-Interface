import pickle
import pathlib
import shutil
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import requests
import seaborn as sns
from streamlit_lottie import st_lottie
import os
import wfdb
from final_project import *
from feature_extraction import *

st.set_page_config(page_title='ECG Based Authentication Interface',
                   page_icon=':star:', layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("## Classification")

# region Load Model
non_fid_model = pickle.load(open('random_forest_classifier_nonFid.pkl', 'rb'))
non_fid_bonus_model = pickle.load(
    open('random_forest_classifier_nonFidBonus.pkl', 'rb'))
fid_model = pickle.load(open('random_forest_classifier_Fid.pkl', 'rb'))
# endregion

# region reading Data from Function


def read_data(path):
    fs = 1000
    patient = wfdb.rdrecord(path, channels=[1])
    signal = patient.p_signal[:, 0]
    time = len(signal) / fs
    return signal, time, patient
# endregion


# region Load Data
uploaded_files = st.file_uploader(
    "Choose Signal: ", accept_multiple_files=True)

os.makedirs("temp", exist_ok=True)
signal_name = ""
signal = ""
time = ""
for uploaded_file in uploaded_files:
    signal_name = uploaded_file.name.split(".")[0]
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    # st.write("Done Reading file:", uploaded_file.name)
if signal_name != "":
    signal, time, record = read_data("temp/"+signal_name)
    # st.write("Signal Shape:", signal.shape)
    fig = wfdb.plot_wfdb(record=record, figsize=(200, 20), title='Record')
    st.pyplot(fig)
    shutil.rmtree("temp")
    st.write("Done Reading files!")

# endregion
methods = {'Fiducial': 0, 'Non Fiducial': 1, 'Non Fiducial Bonus': 2}
selected_feature_extraction_method = st.radio('Select Feature Extraction Method: ', [
    "Fiducial", "Non Fiducial", "Non Fiducial Bonus"])
index = methods[selected_feature_extraction_method]
st.write("Selected Feature Extraction Method: ",
         selected_feature_extraction_method)


input_signal = None
# region preprocessing
if signal != "" and time != "":
    input_signal = processing(signal)
# endregion


def get_features(signal, index):
    if index == 0:
        return Fiducial_Features([signal])
    elif index == 1:
        return non_fiducial_features([signal])
    elif index == 2:
        return non_fiducial_features_bonus_preprocessing([signal])


# region Classification
if st.button("Predict"):
    if index == 0:
        features, values = get_features(input_signal, index)
        values = np.array(values).reshape(-1, 23)
        prediction = fid_model.predict_proba(values[:, :22])
        threshold_percentage = 0.95
        flag = 0
        for i in range(0, len(prediction)):
            for subject_id, percentage in enumerate(prediction[i]):

                if percentage >= threshold_percentage:
                    st.write(
                        f"Identified as subject {subject_id+1} with {percentage*100}% certainty.")
                    flag = 1
            if flag == 1:
                break

        if flag == 0:
            st.write("subject is undefind")
    elif index == 1:
        features, values = get_features(input_signal, index)
        values = np.array(values).reshape(1, 81)
        prediction = non_fid_model.predict_proba(values[:, :80])

        threshold_percentage = 0.5
        flag = 0
        for i in range(0, len(prediction)):
            for subject_id, percentage in enumerate(prediction[i]):

                if percentage >= threshold_percentage:
                    st.write(
                        f"Identified as subject {subject_id+1} with {percentage*100}% certainty.")
                    flag = 1
        if flag == 0:
            st.write("subject is undefind")

    elif index == 2:
        features, values = get_features(input_signal, index)
        values = np.array(values).reshape(-1, 41)
        prediction = non_fid_bonus_model.predict_proba(values[:, :40])
        threshold_percentage = 0.95
        flag = 0
        for i in range(0, len(prediction)):
            for subject_id, percentage in enumerate(prediction[i]):

                if percentage >= threshold_percentage:
                    st.write(
                        f"Identified as subject {subject_id+1} with {percentage*100}% certainty.")
                    flag = 1
            if flag == 1:
                break
        if flag == 0:
            st.write("subject is undefind")
# endregion
