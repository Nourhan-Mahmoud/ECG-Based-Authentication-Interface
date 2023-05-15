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
from FeatureExtraction_NB import read_data, Feature_Detection
import wfdb

st.set_page_config(page_title='ECG Based Authentication Interface',
                   page_icon=':star:', layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("## Classification")

# region Load Model
fid_svm = pickle.load(open('HCI_SVM_F.pkl', 'rb'))
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


# region Feature Extraction
fiducial_points = ""
non_fiducial_points = ""
if signal != "" and time != "":
    fiducial_points, non_fiducial_points = Feature_Detection(signal)
# endregion

# region Classification
if st.button("Predict"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("With Fiducial Features")
        st.write("---")
        prediction = fid_svm.predict(fiducial_points)
        values, counts = np.unique(prediction, return_counts=True)
        st.write("Person ID:", int(values[counts.argmax()]))
        st.write("Prediction For Each Segment:")
        st.table(prediction)
    with col2:
        st.subheader("With Non Fiducial Features")
        st.write("---")
# endregion
