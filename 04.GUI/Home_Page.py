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


def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.write("## ECG Based Authentication Interface")
st.write("---")

animation = load_lottie(
    'https://assets8.lottiefiles.com/packages/lf20_zw7jo1.json')
with st.container():
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(
            """
            Authentication is an important factor to manage security. 
            The traditional authentication methods are not secure enough to protect the user’s data. 
            So, Using ECG signals as a biometric authentication method is a good solution to solve this problem.   
            ### Team Members
            - Ahmed Mohamed Samy 
            - Ayman Hassan
            - Elmoatazbellah Ahmed
            - Nora Ekramy
            - Nourhan Mahmoud
            ### See More
            - GitHub Repository: [Project](https://github.com/Nourhan-Mahmoud/ECG-Based-Authentication-Interface)
            """
        )
    with right_col:
        st_lottie(animation, height=500, width=500,
                  key='Signature Recognition')
