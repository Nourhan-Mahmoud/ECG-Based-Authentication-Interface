

import wfdb
from scipy import signal
import scipy
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
import pywt
from feature_extraction import processing, Fiducial_Points_Detection, nonFiducial, non_fiducial_features_bonus
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


def load_subjects():
    fs = 1000
    # best start = 1300, best end = 3200
    patient_1 = wfdb.rdrecord('..//01.Dataset/104/s0306lre', channels=[1])
    patient_2 = wfdb.rdrecord('..//01.Dataset/117/s0291lre', channels=[1])
    patient_3 = wfdb.rdrecord('..//01.Dataset/122/s0312lre', channels=[1])
    patient_10 = wfdb.rdrecord('..//01.Dataset/166/s0275lre', channels=[1])
    patient_5 = wfdb.rdrecord('..//01.Dataset/173/s0305lre', channels=[1])
    patient_6 = wfdb.rdrecord('..//01.Dataset/182/s0308lre', channels=[1])
    patient_7 = wfdb.rdrecord('..//01.Dataset/234/s0460_re', channels=[1])
    patient_8 = wfdb.rdrecord('..//01.Dataset/238/s0466_re', channels=[1])
    patient_9 = wfdb.rdrecord('..//01.Dataset/255/s0491_re', channels=[1])
    patient_4 = wfdb.rdrecord('..//01.Dataset/252/s0487_re', channels=[1])

    signal_1 = processing(patient_1.p_signal[:, 0])
    signal_2 = processing(patient_2.p_signal[:, 0])
    signal_3 = processing(patient_3.p_signal[:, 0])
    signal_4 = processing(patient_4.p_signal[:, 0])
    signal_5 = processing(patient_5.p_signal[:, 0])
    signal_6 = processing(patient_6.p_signal[:, 0])
    signal_7 = processing(patient_7.p_signal[:, 0])
    signal_8 = processing(patient_8.p_signal[:, 0])
    signal_9 = processing(patient_9.p_signal[:, 0])
    signal_10 = processing(patient_10.p_signal[:, 0])
    time = len(signal_1[0]) / fs
    return [signal_1, signal_2, signal_3, signal_4, signal_5, signal_6, signal_7, signal_8, signal_9, signal_10]
# # Preprocessing


def Fiducial_Features(signals):
    fid_features = []
    fid_features_values = []
    for i, signal in enumerate(signals):
        fid_feature = Fiducial_Points_Detection(signal)
        fid_feature["class"] = i+1
        fid_features_values.append(fid_feature.values)
        fid_features.append(fid_feature)
    return fid_features, fid_features_values


def create_data_frame_fid(fid_features_values):
    df = pd.DataFrame({})
    for i in range(len(7)):
        df = np.concatenate((df, fid_features_values[i]), axis=0)
    np.save("fiducial_feature.npy", df)
    return df


# # Fiducial Model
#Fid_Data = np.load("fiducial_feature.npy")
#X_train = Fid_Data[:, :22]
#y_train = Fid_Data[:, -1]


# random forest classifier
def random_forest_classifier(X_train, y_train, filename):
    random_forest_classifier_object = RandomForestClassifier(
        n_estimators=100, random_state=42)
    random_forest_classifier_object.fit(X_train, y_train)
    # save
    pickle.dump(random_forest_classifier_object, open(filename, 'wb'))

# SVM


def SVM_classifier(X_train, y_train, filename):
    SVM_classifier_object = svm.SVC(kernel='linear')
    SVM_classifier_object.fit(X_train, y_train)
    # save
    pickle.dump(SVM_classifier_object, open(filename, 'wb'))
# LR


def LR_classifier(X_train, y_train, filename):
    LR_classifier_object = LogisticRegression(random_state=0)
    LR_classifier_object.fit(X_train, y_train)
    # save
    pickle.dump(LR_classifier_object, open(filename, 'wb'))


# # Non_feducial feature
def data_frame_non(non_fid, label):
    df = pd.DataFrame({})
    for i in range(len(non_fid)):
        df[i] = non_fid[i]
    df['class'] = label
    return df


def non_fiducial_features(signals):
    non_fid_features = []
    non_fid_features_values = []
    for i, signal in enumerate(signals):
        non_fid_feature = nonFiducial(signal).reshape(80, 1)
        non_fid_feature = data_frame_non(non_fid_feature, i+1)
        non_fid_features_values.append(non_fid_feature.values)
        non_fid_features.append(non_fid_feature)
    return non_fid_features, non_fid_features_values


def create_data_frame_non_fid(non_fid_features_values):
    df = pd.DataFrame({})
    for i in range(len(7)):
        df = np.concatenate((df, non_fid_features_values[i]), axis=0)
    np.save("non_fiducial_feature.npy", df)
    df.to_csv("Non_feaducial_feature.csv", index=False)
    return df


'''
X_train = Data_Non_fid.drop('class', axis=1)
y = Data_Non_fid['class']


threshold_percentage = 0.5
flag = 0
for i in range(0, len(probs)):
    for subject_id, percentage in enumerate(probs[i]):

        if percentage >= threshold_percentage:
            print(
                f"Identified as subject {subject_id+1} with {percentage}% certainty.")
            flag = 1
        # else:
        #     print(f"Identified as subject {subject_id+1} , but with {percentage}% certainty below the threshold.")

if flag == 0:
    print("subject is undefind")

'''
# # Bouns


def data_frame(non_fid, label):
    df = pd.DataFrame({})
    for i in range(len(non_fid)):
        df[i] = non_fid[i]
    df = df.T
    df['class'] = label
    return df


def non_fiducial_features_bonus_preprocessing(signals):
    non_fid_features = []
    non_fid_features_values = []
    for i, signal in enumerate(signals):
        non_fid_feature = non_fiducial_features_bonus(signal)
        non_fid_feature = data_frame(non_fid_feature, i+1)
        non_fid_features_values.append(non_fid_feature.values)
        non_fid_features.append(non_fid_feature)
    return non_fid_features, non_fid_features_values


def create_data_frame_non_fid_bonus(non_fid_features_values):
    df = pd.DataFrame({})
    for i in range(len(7)):
        df = np.concatenate((df, non_fid_features_values[i]), axis=0)
    np.save("bouns_feature.npy", df)
    df.to_csv("bouns_feature.csv", index=False)
    return df


'''
threshold_percentage = 0.95
flag = 0
for i in range(0, len(probs)):
    for subject_id, percentage in enumerate(probs[i]):

        if percentage >= threshold_percentage:
            print(
                f"Identified as subject {subject_id+1} with {percentage}% certainty.")
            flag = 1
    if flag == 1:
        break

if flag == 0:
    print("subject is undefind")
'''
