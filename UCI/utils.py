#!/usr/bin/env python
# coding: utf-8

# This file contains utility functions that are used across multiple parts of your program, including data manipulation, file I/O operations, or other helper functions.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io
import copy
import random
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")



class build_data:
    def __init__(self, path, num_sensor):
        self.path = path
        self.num_sensor = num_sensor
        self.file_lst = os.listdir(self.path)
        self.files = {}
        
    def import_data(self):
        
        for i in range(len(self.file_lst)):
            file_file_lst = os.listdir(self.path+self.file_lst[i])
            for j in range(len(file_file_lst)):
                try:
                    self.files['f{}_{}'.format(i,j)] = pd.read_csv(self.path+self.file_lst[i]+'/'+file_file_lst[j], delimiter='\t')   
                except:
                    raise IOError(f'Invalid file path')
        
        check_len=[]
        
        cnt=0
        for idx, item in enumerate(self.file_lst):
            f_lst = os.listdir(self.path+item)
            for j in range(len(f_lst)):        
                  
                for k in range(len(self.files['f{}_{}'.format(idx,j)])-1):
                    if self.files['f{}_{}'.format(idx,j)]['class'][k] == self.files['f{}_{}'.format(idx,j)]['class'][k+1]:
                        cnt+=1
                        
                    elif self.files['f{}_{}'.format(idx,j)]['class'][k] != self.files['f{}_{}'.format(idx,j)]['class'][k+1]: #stop     
                        check_len.append(cnt)
                        cnt=0   
        
                    else:
                        raise AttributeError(f"Invalid file type.")
        
        return min(check_len)

    
    
    def data_processing(self):
        df=[]
        SENSOR={'ch{}'.format(sensors): [] for sensors in range(1, self.num_sensor+1)}
        min_num = self.import_data()
        for i, item in enumerate(self.file_lst):
            f_lst = os.listdir(self.path+item)
            for j in range(len(f_lst)):        
                for k in range(len(self.files['f{}_{}'.format(i,j)])-1):
                    if int(self.files['f{}_{}'.format(i,j)]['class'][k]) == 7 or int(self.files['f{}_{}'.format(i,j)]['class'][k]) == 8: pass
                    
                    elif self.files['f{}_{}'.format(i,j)]['class'][k] == self.files['f{}_{}'.format(i,j)]['class'][k+1]:
                        for sensors in range(1,self.num_sensor+1):
                            SENSOR['ch{}'.format(sensors)].append(self.files['f{}_{}'.format(i,j)]['channel{}'.format(sensors)][k])
                        
                    elif self.files['f{}_{}'.format(i,j)]['class'][k] != self.files['f{}_{}'.format(i,j)]['class'][k+1]: #stop     
                        selected_idx = np.sort(random.sample(np.arange(len(SENSOR['ch1'])).tolist(), min_num))
                        
                        for sensors in range(1,self.num_sensor+1):
                            SENSOR['ch{}_selected'.format(sensors)]=[]
                            for r in selected_idx:
                                SENSOR['ch{}_selected'.format(sensors)].append(SENSOR['ch{}'.format(sensors)][r])
                            SENSOR['ch{}_selected'.format(sensors)].append(self.files['f{}_{}'.format(i,j)]['class'][k])
                            df.append(SENSOR['ch{}_selected'.format(sensors)])
                            SENSOR['ch{}'.format(sensors)]=[] 
                    else:
                        raise AttributeError(f"Invalid file type.")
        
        return pd.DataFrame(df), min_num


    def data_save(self):
        
        df, min_num = self.data_processing()
        try:
            labels = [int(label) for label in df[min_num]]
            df['Label'] = labels
            del df[min_num]
            
        except (ValueError, KeyError, IndexError) as e:
            print("An error occurred:", e)
        
        if df.isnull().values.any() == True:
            raise IndexError(f"Incorrect index")
        elif df.isnull().values.any() == False:
            pass
        
        label_counts = {i: 0 for i in range(self.num_sensor)}
        
        for label_val in df['Label']:
            if label_val in label_counts:
                label_counts[label_val] += 1
            
        zero_alignments = [i for i in range(0, label_counts[0], self.num_sensor)]
        selected_zeros = np.sort(random.sample(zero_alignments, int(label_counts[1] / self.num_sensor)))
        
        processed_data = []
        
        for label_type, group_data in df.groupby("Class"):
            if label_type == 0:
                for i in range(len(selected_zeros)):
                    for j in range(self.num_sensor):
                        processed_data.append(group_data.iloc[i + j, :].tolist())
        
                processed_df = pd.DataFrame(processed_data)
                label_values = copy.deepcopy(processed_df[min_num])
                del processed_df[min_num]
                processed_df['Label'] = label_values
                
            else:
                processed_df = pd.concat([processed_df, group_data])
        
        processed_df = processed_df.reset_index(drop=True)
        
        processed_df = processed_df[processed_df['Label'] != 0]
        processed_df['Label'] = processed_df['Label']-1

        try:
            processed_df.to_csv('./UCI_emg_gesture_dataset.csv', index=False)
        except PermissionError:
            print("Permission denied: Unable to save the DataFrame to CSV.")
        except FileNotFoundError:
            print("File not found: Unable to save the DataFrame to CSV.")
        except Exception as e:
            print(f"An error occurred: {e}")



def call_data(path):
    
    #path = 'C:/Users/hml76/Desktop/Jupyter/Paper1__renew/Github/UCI/Dataset/'

    dataset = pd.read_csv(path+'UCI_emg_gesture_dataset.csv')
    dataset = dataset[dataset['Label'] != 0]
    dataset['Label'] = dataset['Label']-1

    if dataset.isnull().values.any(): 
        raise ValueError("DataFrame contains no missing values.")  
    else: 
        pass

    return dataset

    
def data_split(df, ratio): 
    
    Train_data, Test_data=[], []

    for _, label_groups in df.groupby("Class"):
        indices = np.random.rand(len(label_groups.index)) <= (1-ratio)  #0.8
        Train_data.append(label_groups[indices])
        Test_data.append(label_groups[~indices])

    Train_data = pd.concat(Train_data).sample(frac=1)
    Test_data = pd.concat(Test_data).sample(frac=1)
    
    feature_names = list(set(df.columns)-{"Class"})
    print(f"Train data shape: {Train_data.shape}\nTest data shape: {Test_data.shape}")

    X_train = Train_data[feature_names].index.to_numpy()
    y_train = Train_data["Class"].astype(int)
    X_test = Test_data[feature_names].index.to_numpy()
    y_test = Test_data["Class"].astype(int)
    
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


