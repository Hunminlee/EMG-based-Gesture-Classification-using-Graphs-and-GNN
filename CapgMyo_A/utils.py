#!/usr/bin/env python
# coding: utf-8

# This file contains utility functions that are used across multiple parts of your program, including data manipulation, file I/O operations, or other helper functions.


import gc
import glob
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



class data_processing:
    def __init__(self, num_sensor, subject_idx, num_labels, num_gesture):
        self.num_sensor = num_sensor
        self.subject = subject_idx
        self.num_gesture = num_gesture
        self.num_label = num_labels 
        
    
    def build_data(self, path, D_Type):    
        #val_lst, labels = [], []
        #length=self.normalize_sampling(path) #BandMyo; dynamic length

        path = f'C:/Users/hml76/Desktop/Jupyter/Paper1__renew/CapgMyo/Data/DB_{D_Type}_preproceesed/'     
        file_list = os.listdir(path)

        num_gesture = 8
        num_trial = 10
        num_sensor = 128
        length = 1000  #CapgMyo 
        total_data, label_lst = [], []
        idx=1

        if D_Type == 'A':
            try:
                if self.subject < 10:
                    for gesture in range(1,num_gesture+1):
                        for trial in range(1,num_trial+1): #trial 1~10
                            if trial < 10:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/00{}-00{}-00{}.mat'.format(self.subject, gesture, trial)
                                    )['data'].T
                                for idx in range(1):  
                                    for sensor in range(num_sensor):                       
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                            else:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/00{}-00{}-010.mat'.format(self.subject, gesture)
                                    )['data'].T
                                for idx in range(1):  
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                
                elif self.subject >= 10:
                    for gesture in range(1,num_gesture+1):
                        for trial in range(1,num_trial+1): 
                            if trial < 10:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/0{}-00{}-00{}.mat'.format(self.subject, gesture, trial)
                                    )['data'].T
                                for idx in range(1):  
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                            else:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/0{}-00{}-010.mat'.format(self.subject, gesture)
                                    )['data'].T
                                for idx in range(1): 
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                else:
                    raise TypeError("Incorrect subject index")
            except:
                raise TypeError("Incorrect file name")


        elif D_Type == 'B':
            try:
                if self.subject < 10:
                    for gesture in range(1,num_gesture+1):
                        for trial in range(1,num_trial+1): 
                            if trial < 10:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/00{}-00{}-00{}.mat'.format(self.subject, gesture, trial)
                                    )['data'].T
                                for idx in range(1):  
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                                
                            else:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/00{}-00{}-010.mat'.format(self.subject, gesture)
                                    )['data'].T
                                for idx in range(1): 
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                                
                elif self.subject >= 10 and self.subject < 20:
                    for gesture in range(1,num_gesture+1):
                        for trial in range(1,num_trial+1): 
                            if trial < 10:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/0{}-00{}-00{}.mat'.format(self.subject, gesture, trial)
                                    )['data'].T
                                for idx in range(1): 
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                                
                            else:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/0{}-00{}-010.mat'.format(self.subject, gesture)
                                    )['data'].T
                                for idx in range(1): 
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                                
                elif self.subject >= 20:
                    for gesture in range(1,num_gesture+1):
                        for trial in range(1,num_trial+1): 
                            if trial < 10:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/0{}-00{}-00{}.mat'.format(self.subject, gesture, trial)
                                    )['data'].T
                                for idx in range(1): 
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                                
                            else:
                                data = scipy.io.loadmat(
                                    path + file_list[self.subject-1] + '/0{}-00{}-010.mat'.format(self.subject, gesture)
                                    )['data'].T
                                for idx in range(1): 
                                    for sensor in range(num_sensor):                        
                                        total_data.append(data[sensor][length*idx:length*(idx+1)].tolist())
                                        label_lst.append(gesture-1)
                                
                else:
                    raise TypeError("Incorrect subject index")
            except:
                raise TypeError("Incorrect file name")
                        
        df = pd.DataFrame(total_data)
        df['Label'] = label_lst
        
        return df
    
    
    def data_split(self, df, ratio):
        
        Train_data, Test_data=[], []

        for _, label_groups in df.groupby("Label"):
            indices = np.random.rand(len(label_groups.index)) <= (1-ratio)  #0.8
            Train_data.append(label_groups[indices])
            Test_data.append(label_groups[~indices])

        Train_data = pd.concat(Train_data).sample(frac=1)
        Test_data = pd.concat(Test_data).sample(frac=1)
        
        feature_names = list(set(df.columns)-{"Label"})
        print(f"Train data shape: {Train_data.shape}\nTest data shape: {Test_data.shape}")

        X_train = Train_data[feature_names].index.to_numpy()
        y_train = Train_data["Label"]
        X_test = Test_data[feature_names].index.to_numpy()
        y_test = Test_data["Label"]
        
        node_features = tf.cast(df[feature_names].to_numpy(), dtype=tf.dtypes.float32)
        
        return node_features, X_train, y_train, X_test, y_test


def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()


