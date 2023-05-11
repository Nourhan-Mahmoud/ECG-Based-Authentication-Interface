import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import requests
import seaborn as sns
from streamlit_lottie import st_lottie

st.set_page_config(page_title='ECG Based Authentication Interface',
                   page_icon=':star:', layout="wide")

st.write("## Classification")

# region Load Data
uploaded_files = st.file_uploader(
    "Choose Signal: ", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
# endregion

# region Pre-processing

# endregion

# region Segmentation

# endregion

# region Feature Extraction

# endregion

# region Load Model

# endregion

# region Classification

# endregion
