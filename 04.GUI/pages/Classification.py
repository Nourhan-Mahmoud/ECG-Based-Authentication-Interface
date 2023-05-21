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
    fig = wfdb.plot_wfdb(
        record=record, figsize=(200, 20), title='Record')
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


signal1, signal3, signal2, signal4 = None, None, None, None
# region preprocessing
if signal != "" and time != "":
    signal1 = processing(signal)
    signal2 = processing(signal)
    signal3 = processing(signal)
    signal4 = processing(signal)
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
    col1, col2 = st.columns(2)
    with col1:
        if index == 0:
            features, values = get_features(signal1, index)
            values = np.array(values).reshape(-1, 23)
            prediction = fid_model.predict_proba(values[:, :22])
            threshold_percentage = 0.8
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
            features, values = get_features(signal2, index)
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
            features, values = get_features(signal3, index)
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

    with col2:
        if index == 0:
            samp_start = 500
            samp_end = 2000
            res = points_for_plot(signal4, start=samp_start, end=samp_end)
            fs = 1000
            time = len(res['denoised_signal'])/fs
            ts = np.arange(0, time, 1.0 / fs)
            fig, ax = plt.subplots()
            ax.plot(ts, res['denoised_signal'],
                    alpha=0.6, lw=1, label="Raw signal")
            ax.scatter(res['qx']/fs, res['qy'], alpha=0.5,
                       color='red', label="Q point")
            ax.scatter(res['sx']/fs, res['sy'], alpha=0.5,
                       color='green', label="S point")
            ax.scatter(res['Rx']/fs, res['Ry'], alpha=0.5,
                       color='blue', label="R point")
            ax.scatter(res['qrs_on_x']/fs, res['qrs_on_y'],
                       alpha=0.5, color='black', label="QRS onset")
            ax.scatter(res['qrs_off_x']/fs, res['qrs_off_y'],
                       alpha=0.5, color='yellow', label="QRS offset")
            ax.scatter(res['Px']/fs, res['Py'], alpha=0.5,
                       color='orange', label="P point")
            ax.scatter(res['p_on_x']/fs, res['p_on_y'],
                       alpha=0.5, color='pink', label="P onset")
            ax.scatter(res['p_off_x']/fs, res['p_off_y'],
                       alpha=0.5, color='cyan', label="P offset")
            ax.scatter(res['Tx']/fs, res['Ty'], alpha=0.5,
                       color='purple', label="T point")
            ax.scatter(res['t_on_x']/fs, res['t_on_y'],
                       alpha=0.5, color='brown', label="T onset")
            ax.scatter(res['t_off_x']/fs, res['t_off_y'],
                       alpha=0.5, color='gray', label="T offset")
            ax.set_title("Fiducial Points")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (mV)")
            ax.legend(prop={"size": 7}, loc="upper right")
            st.pyplot(fig)
        elif index == 1:
            cmp = non_fid_for_plot(signal2)
            fig, ax = plt.subplots(5)
            row = 0
            for i in range(0, 5):
                ax[row].plot(cmp[i])
                ax[row].set_title("")
                row += 1
            for ax in fig.get_axes():
                ax.label_outer()
            fig.tight_layout(h_pad=0)
            st.pyplot(fig)
        elif index == 2:
            v1 = non_fiducial_features_bonus_plots(signal3)
            v2 = non_fiducial_features_bonus_plots2(signal3)
            fig, ax = plt.subplots(2)
            row = 0
            for i in range(0, 2):
                if i == 0:
                    ax[row].plot(v2[0])
                    ax[row].set_title("")
                    row += 1
                else:
                    ax[row].plot(v1[0])
                    ax[row].set_title("")
                    row += 1
            for ax in fig.get_axes():
                ax.label_outer()
            fig.tight_layout(h_pad=0)
            st.pyplot(fig)
# endregion
