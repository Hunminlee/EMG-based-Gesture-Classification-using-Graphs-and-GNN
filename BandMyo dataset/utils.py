#!/usr/bin/env python
# coding: utf-8

# This file contains utility functions that are used across multiple parts of your program, including data manipulation, file I/O operations, or other helper functions.


import gc
import glob
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class data_processing:
    def __init__(self, num_sensor, num_labels, num_gesture):
        self.num_sensor = num_sensor
        self.num_gesture = num_gesture
        self.num_label = num_labels 
        self.window_size = 150
        self.total_subs = 6

    def normalize_sampling(self, path):
        lst_len=[]
        digit_ch=10
        digit_new=self.num_label-digit_ch
        
        for s_idx in range(self.total_subs):
            for label in range(1,digit_ch):   
                try:
                    files = glob.glob(path + '/00{}/00{}/*.mat'.format(s_idx, label))
                    for gesture in files:
                        data = scipy.io.loadmat(gesture)['data'].T
                        lst_len.extend([len(data[i]) for i in range(self.num_sensor)])
                except:
                    raise IOError(f"Incorrect file or path for file idx {s_idx} - {label}")    
            
            for label in range(digit_new+1):  
                try:
                    files = glob.glob(path + '/00{}/01{}/*.mat'.format(s_idx, label))
                    for gesture in files:
                        data = scipy.io.loadmat(gesture)['data'].T
                        lst_len.extend([len(data[i]) for i in range(self.num_sensor)])
                except IOError:
                    raise IOError(f"Incorrect file or path for file idx {s_idx} - {label}")    
                #except PermissionError:
                #    print("Permission denied: Unable to save the DataFrame to CSV.")
                #except FileNotFoundError:
                #    print("File not found: Unable to save the DataFrame to CSV.")
                #except Exception as e:
                #    print(f"An error occurred: {e}")

        return np.min(lst_len)
        
    
    def build_data(self, path):    
        
        #length=self.normalize_sampling(path)
        data, dataset, labels = {}, [], []
        for s_idx in range(self.total_subs):
            for label in range(1,self.num_label+1):  
                try:
                    if label < 10:
                        f_lst = glob.glob(path + '/00{}/00{}/*.mat'.format(s_idx, label))
                    else:
                        f_lst = glob.glob(path + '/00{}/0{}/*.mat'.format(s_idx, label))
                    
                    for gesture in range(len(f_lst)): 
                        data['emg_sub{}_label{}_Gesture{}'.format(s_idx, label, gesture)] = scipy.io.loadmat(f_lst[gesture])['data'].T
                except:
                    raise IOError(f"Incorrect file or path for file idx {s_idx} - {label}")    

            for label in range(1,self.num_label+1):
                for gesture in range(self.num_gesture):
                    lens = int(len(data['emg_sub{}_label{}_Gesture{}'.format(s_idx, label, gesture)][0])/self.window_size)
                    for idx in range(lens):     
                        for sensor_idx in range(self.num_sensor):
                            dataset.append(
                                data['emg_sub{}_label{}_Gesture{}'.format(s_idx, label, gesture)][sensor_idx][self.window_size*idx:self.window_size*(idx+1)]
                                )
                            labels.append(label-1)

        df = pd.DataFrame(dataset)
        df['Label'] = labels
        
        return df
    
    
    def data_split(self, df, ratio):
        
        Train_data, Test_data=[], []

        for _, label_groups in df.groupby("Label"):
            indices = np.random.rand(len(label_groups.index)) <= ratio  #0.8
            Train_data.append(label_groups[indices])
            Test_data.append(label_groups[~indices])

        Train_data = pd.concat(Train_data).sample(frac=1)
        Test_data = pd.concat(Test_data).sample(frac=1)
        
        feature_names = list(set(df.columns) - {"Label"})
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



