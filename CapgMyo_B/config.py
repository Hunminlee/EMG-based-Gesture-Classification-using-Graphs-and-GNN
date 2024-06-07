#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf

class Graph_CapgMyo:
    def __init__(self, df, num_sensor):
        self.df = df
        self.num_sensor = num_sensor
    
    def generate_graph(self, src, dst):
        graph = pd.DataFrame([src, dst]).T
        graph.columns = ['source', 'target']

        edges = graph[["source", "target"]].to_numpy().T
        edge_weights = tf.ones(shape=edges.shape[1])

        return graph, edges, edge_weights

    def KNN1(self):
        row_sensors = 8
        edge = {}
        for i in range(row_sensors):
            if i==0: edge['lst_{}'.format(i)]=[1,row_sensors,row_sensors+1,self.num_sensor-row_sensors,self.num_sensor-row_sensors+1]
            elif i==(row_sensors-1): edge['lst_{}'.format(i)]=[row_sensors-2,(row_sensors*2)-2,(row_sensors*2)-1,self.num_sensor-2,self.num_sensor-1]
            else: edge['lst_{}'.format(i)]=[i-1, i+1, i+(row_sensors-1), i+row_sensors, i+row_sensors+1, i+(self.num_sensor-row_sensors-1), 
                              i+(self.num_sensor-row_sensors), i+(self.num_sensor-row_sensors+1)]
                
        for i in range(row_sensors, self.num_sensor-row_sensors):
            if i%row_sensors == 0: edge['lst_{}'.format(i)]=[i-row_sensors, i-(row_sensors-1), i+1, i+row_sensors, i+(row_sensors+1)]
            elif i%row_sensors == (row_sensors-1): edge['lst_{}'.format(i)]=[i-row_sensors, i-(row_sensors+1), i-1, i+(row_sensors-1), i+row_sensors]
            else: edge['lst_{}'.format(i)]=[i-(row_sensors+1), i-row_sensors, i-(row_sensors-1), i-1, i+1, i+(row_sensors-1), i+row_sensors, i+(row_sensors+1)]
        
        for i in range(self.num_sensor-row_sensors,self.num_sensor):
            if i==(self.num_sensor-row_sensors): edge['lst_{}'.format(i)]=[i-row_sensors,i-(row_sensors-1),i+1,0,1]
            elif i==(self.num_sensor-1): edge['lst_{}'.format(i)]=[i-(row_sensors+1),i-row_sensors,i-1,row_sensors-2,row_sensors-1]
            else: edge['lst_{}'.format(i)]=[i-1, i-(row_sensors+1), i-row_sensors, i-(row_sensors-1), i+1, i-(self.num_sensor-row_sensors+1), 
                              i-(self.num_sensor-row_sensors), i-(self.num_sensor-row_sensors-1)]
        
        src_tmp,tgt_tmp = [],[]

        for sensors in range(self.num_sensor):
            for item in edge['lst_{}'.format(sensors)]:
                src_tmp.append(sensors)
                tgt_tmp.append(item)
        src,tgt=[],[]
        src = src + src_tmp
        tgt = tgt + tgt_tmp

        for _ in range(int(len(self.df)/self.num_sensor)-1):
            for idx in range(len(src_tmp)):
                src_tmp[idx] = src_tmp[idx] + self.num_sensor
                tgt_tmp[idx] = tgt_tmp[idx] + self.num_sensor
            src = src + src_tmp
            tgt = tgt + tgt_tmp
            
        return self.generate_graph(src, tgt)
    
    def KNN1_SW(self): 
        edge = {}
        row_sensors = 8
        for i in range(row_sensors,int(self.num_sensor-row_sensors)):
            if i%row_sensors == 0: edge['lst_{}'.format(i)]=[i-row_sensors, i-row_sensors+1, i+1, i+row_sensors, i+row_sensors+1]
            elif i%row_sensors == (row_sensors-1): edge['lst_{}'.format(i)]=[i-row_sensors, i-row_sensors-1, i-1, i+row_sensors-1, i+row_sensors]
            else: edge['lst_{}'.format(i)]=[i-row_sensors-1, i-row_sensors, i-row_sensors+1, i-1, i+1, i+row_sensors-1, i+row_sensors, i+row_sensors+1]

        for i in range(row_sensors):
            if i==0: edge['lst_{}'.format(i)]=[1,row_sensors,row_sensors+1,self.num_sensor-row_sensors,self.num_sensor-row_sensors+1]
            elif i==(row_sensors-1): edge['lst_{}'.format(i)]=[row_sensors-2,row_sensors*2-2,row_sensors*2-1,self.num_sensor-2,self.num_sensor-1]
            else: edge['lst_{}'.format(i)]=[i-1, i+1, i+row_sensors-1, i+row_sensors, i+row_sensors+1, i+self.num_sensor-row_sensors-1, 
                                            i+self.num_sensor-row_sensors, i+self.num_sensor-row_sensors+1]

        for i in range(int(self.num_sensor-row_sensors),self.num_sensor):
            if i==int(self.num_sensor-row_sensors): edge['lst_{}'.format(i)]=[i-row_sensors,i-row_sensors+1,i+1,0,1]
            elif i==(self.num_sensor-1): edge['lst_{}'.format(i)]=[i-row_sensors-1,i-row_sensors,i-1,row_sensors-2,row_sensors-1]
            else: edge['lst_{}'.format(i)]=[i-1, i-row_sensors-1, i-row_sensors, i-row_sensors+1, i+1, i-(self.num_sensor-row_sensors+1), 
                                            i-(self.num_sensor-row_sensors), i-(self.num_sensor-row_sensors-1)]

        src_tmp,tgt_tmp = [],[]
        num_cnt = 0

        for sensors in range(self.num_sensor):
            for i in range(len(edge['lst_{}'.format(sensors)])):
                src_tmp.append(sensors)
                tgt_tmp.append(edge['lst_{}'.format(sensors)][i])

        src,tgt = [],[]

        src = src + src_tmp 
        tgt = tgt + tgt_tmp

        for i in range(int(len(self.df)/self.num_sensor)-1):
            for idx in range(len(src_tmp)):
                src_tmp[idx] = src_tmp[idx] + self.num_sensor
                tgt_tmp[idx] = tgt_tmp[idx] + self.num_sensor

            src = src + src_tmp
            tgt = tgt + tgt_tmp
                
            for k in range(self.num_sensor):  
                tgt.append((i+1)*self.num_sensor + k)  
                src.append(num_cnt)
                num_cnt = num_cnt + 1
        
        return self.generate_graph(src, tgt)        

    def KNN2(self):
        row_sensors=8
        edge={}

        for i in range(row_sensors,int(self.num_sensor-row_sensors)):
            if i%row_sensors == 0: edge['lst_{}'.format(i)]=[i-row_sensors, i-row_sensors+1, i+1, i+row_sensors, i+row_sensors+1]
            elif i%row_sensors == (row_sensors-1): edge['lst_{}'.format(i)]=[i-row_sensors, i-row_sensors-1, i-1, i+row_sensors-1, i+row_sensors]
            else: edge['lst_{}'.format(i)]=[i-row_sensors-1, i-row_sensors, i-row_sensors+1, i-1, i+1, i+row_sensors-1, i+row_sensors, i+row_sensors+1]

        for i in range(row_sensors):
            if i==0: edge['lst_{}'.format(i)]=[1,row_sensors,row_sensors+1,self.num_sensor-row_sensors,self.num_sensor-row_sensors+1]
            elif i==(row_sensors-1): edge['lst_{}'.format(i)]=[row_sensors-2,row_sensors*2-2,row_sensors*2-1,self.num_sensor-2,self.num_sensor-1]
            else: edge['lst_{}'.format(i)]=[i-1, i+1, i+row_sensors-1, i+row_sensors, i+row_sensors+1, i+self.num_sensor-row_sensors-1, 
                                            i+self.num_sensor-row_sensors, i+self.num_sensor-row_sensors+1]

        for i in range(int(self.num_sensor-row_sensors),self.num_sensor):
            if i==int(self.num_sensor-row_sensors): edge['lst_{}'.format(i)]=[i-row_sensors,i-row_sensors+1,i+1,0,1]
            elif i==(self.num_sensor-1): edge['lst_{}'.format(i)]=[i-row_sensors-1,i-row_sensors,i-1,row_sensors-2,row_sensors-1]
            else: edge['lst_{}'.format(i)]=[i-1, i-row_sensors-1, i-row_sensors, i-row_sensors+1, i+1, i-(self.num_sensor-row_sensors+1), 
                                            i-(self.num_sensor-row_sensors), i-(self.num_sensor-row_sensors-1)]
        
        for i in range(row_sensors):
            if i%row_sensors==0: 
                edge['lst_{}'.format(i)]=[2,row_sensors+2,row_sensors*2,row_sensors*2+1,row_sensors*2+2,self.num_sensor-row_sensors+2,
                                          self.num_sensor-(row_sensors*2),self.num_sensor-(row_sensors*2)+1,self.num_sensor-(row_sensors*2)+2]
            
            elif i%row_sensors==1: 
                edge['lst_{}'.format(i)]=[3,row_sensors+3,row_sensors*2,row_sensors*2+1,row_sensors*2+2,row_sensors*2+3,self.num_sensor-5,self.num_sensor-(2*row_sensors),self.num_sensor-(2*row_sensors)+1,
                            self.num_sensor-(2*row_sensors)+2,self.num_sensor-(2*row_sensors)+3]
            elif i%row_sensors==6: edge['lst_{}'.format(i)]=[4,row_sensors*2-4,row_sensors*2+4,row_sensors*3-3,row_sensors*3-2,row_sensors*3-1,self.num_sensor-4,self.num_sensor-12,
                                               self.num_sensor-11,self.num_sensor-10,self.num_sensor-9]
            elif i%row_sensors==7:edge['lst_{}'.format(i)]=[5,row_sensors*2-3,row_sensors*3-3,row_sensors*3-2,
                                                            row_sensors*3-1,self.num_sensor-3,self.num_sensor-11,self.num_sensor-10,self.num_sensor-9]

        for i in range(row_sensors,int(row_sensors*2)):
            if i%row_sensors==0: edge['lst_{}'.format(i)]=[2,row_sensors+2,row_sensors*2,row_sensors*3,row_sensors*3+1,row_sensors*3+2,self.num_sensor-row_sensors,self.num_sensor-row_sensors+1,self.num_sensor-row_sensors+2]
            elif i%row_sensors==1: edge['lst_{}'.format(i)]=[3,row_sensors+3,row_sensors*2+3,row_sensors*3,row_sensors*3+1,row_sensors*3+2,row_sensors*3+3,self.num_sensor-row_sensors,self.num_sensor-row_sensors+1,
                                               self.num_sensor-row_sensors+2,self.num_sensor-row_sensors+3]
            elif i%row_sensors==6: edge['lst_{}'.format(i)]=[4,row_sensors+4,row_sensors*2+4,row_sensors*3+4,row_sensors*4-3,row_sensors*4-2,row_sensors*4-1,self.num_sensor-4,self.num_sensor-3,self.num_sensor-2,self.num_sensor-1]
            elif i%row_sensors==7: edge['lst_{}'.format(i)]=[5,row_sensors*2-3,row_sensors*3-3,row_sensors*4-3,row_sensors*4-2,row_sensors*4-1,self.num_sensor-3,self.num_sensor-2,self.num_sensor-1]

        for i in range(int(row_sensors*2),int(self.num_sensor-int(row_sensors*2))):
            if i%row_sensors==0:   edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors, i-row_sensors-row_sensors+1, i-row_sensors-row_sensors+2, i-row_sensors+2, i+1+1, 
                                             i+row_sensors+2, i+row_sensors+row_sensors, i+row_sensors+row_sensors+1, i+row_sensors+row_sensors+2]
                
            elif i%row_sensors==1: edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors-1, i-row_sensors-row_sensors, i-row_sensors-row_sensors+1, i-row_sensors-row_sensors+2, 
                                      i-row_sensors+2, i+1+1, i+row_sensors+2, i+row_sensors+row_sensors-1, i+row_sensors+row_sensors, i+row_sensors+row_sensors+1, i+row_sensors+row_sensors+2]
                
            elif i%row_sensors==(row_sensors-2): edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors-2, i-row_sensors-row_sensors-1, i-row_sensors-row_sensors+1, i-row_sensors-row_sensors+2, i-row_sensors-2, 
                                                             i-1-1, i+row_sensors-2, i+row_sensors+row_sensors-2, i+row_sensors+row_sensors-1, i+row_sensors+row_sensors, i+row_sensors+row_sensors+1]
                
            elif i%row_sensors==(row_sensors-1): edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors-2, i-row_sensors-row_sensors-1, i-row_sensors-row_sensors, i-row_sensors-2, i-1-1, i+row_sensors-2, 
                                                             i+row_sensors+row_sensors-2, i+row_sensors+row_sensors-1, i+row_sensors+row_sensors]
                
            else: edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors-2, i-row_sensors-row_sensors-1, i-row_sensors-row_sensors, i-row_sensors-row_sensors+1, i-row_sensors-row_sensors+2, i-row_sensors-2, i-row_sensors+2, 
                              i-1-1, i+1+1, i+row_sensors-2, i+row_sensors+2, i+row_sensors+row_sensors-2, i+row_sensors+row_sensors-1, i+row_sensors+row_sensors, i+row_sensors+row_sensors+1, i+row_sensors+row_sensors+2]

        for i in range(int(self.num_sensor-(2*row_sensors)),int(self.num_sensor-row_sensors)):
            if i%row_sensors==0: edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*4),self.num_sensor-(row_sensors*4)+1,self.num_sensor-(row_sensors*4)+2,
                                             self.num_sensor-(row_sensors*2)-6,self.num_sensor-row_sensors-6,self.num_sensor-6,0,1,2]
            elif i%row_sensors==1: edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*4),self.num_sensor-(row_sensors*4)+1,self.num_sensor-(row_sensors*4)+2,self.num_sensor-(row_sensors*4)+3, 
                                               self.num_sensor-(row_sensors*4)+4,self.num_sensor-(row_sensors*3-3),self.num_sensor-row_sensors-5,self.num_sensor-5,0,1,2,3]
        
            elif i%row_sensors==(row_sensors-2): edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3+4),self.num_sensor-(row_sensors*3+3),self.num_sensor-(row_sensors*3+2),self.num_sensor-(row_sensors*3+1),self.num_sensor-(row_sensors*2+4),self.num_sensor-(row_sensors+4), row_sensors-4, row_sensors-3, row_sensors-2, row_sensors-1]
            elif i%row_sensors==(row_sensors-1): edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3+3),self.num_sensor-(row_sensors*3+2),self.num_sensor-(row_sensors*3+1),self.num_sensor-(row_sensors*2+3), self.num_sensor-(row_sensors+3),self.num_sensor-3,row_sensors-3,row_sensors-2,row_sensors-1]

        for i in range(int(self.num_sensor-row_sensors),self.num_sensor):
            if i%row_sensors==0: edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3),self.num_sensor-(row_sensors*3)+1,self.num_sensor-(row_sensors*3)+2,self.num_sensor-(row_sensors+6),self.num_sensor-6,2,row_sensors,row_sensors+1,row_sensors+2]
            elif i%row_sensors==1: edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3),self.num_sensor-(row_sensors*3)+1,self.num_sensor-(row_sensors*3)+2,self.num_sensor-(row_sensors*3)+3,
                                               self.num_sensor-row_sensors-5,self.num_sensor-5,3,row_sensors,row_sensors+1,row_sensors+2,row_sensors+3]
            elif i%row_sensors==(row_sensors-2): edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3)+4,self.num_sensor-(row_sensors*3)+5,self.num_sensor-(row_sensors*3)+6,self.num_sensor-(row_sensors*3)+7,
                                               self.num_sensor-(row_sensors+4),4,row_sensors+4,row_sensors+5,row_sensors+6,row_sensors+7]
            elif i%row_sensors==(row_sensors-1): edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3)+5,self.num_sensor-(row_sensors*3)+6,self.num_sensor-(row_sensors*3)+7,
                                               self.num_sensor-(row_sensors+5),row_sensors-3,5,row_sensors+5,row_sensors+6,row_sensors+7]

        src_tmp,tgt_tmp = [],[]

        for sensors in range(self.num_sensor):
            for i in range(len(edge['lst_{}'.format(sensors)])):
                src_tmp.append(sensors)
                tgt_tmp.append(edge['lst_{}'.format(sensors)][i])

        src,tgt = [],[]

        src = src + src_tmp 
        tgt = tgt + tgt_tmp

        for i in range(int(len(self.df)/self.num_sensor)-1):
            for idx in range(len(src_tmp)):
                src_tmp[idx] = src_tmp[idx] + self.num_sensor
                tgt_tmp[idx] = tgt_tmp[idx] + self.num_sensor

            src = src + src_tmp
            tgt = tgt + tgt_tmp
            
        return self.generate_graph(src, tgt)

    def KNN2_SW(self):
        row_sensors=8
        edge={}
        num_cnt=0
        for i in range(row_sensors,int(self.num_sensor-row_sensors)):
            if i%row_sensors == 0: edge['lst_{}'.format(i)]=[i-row_sensors, i-row_sensors+1, i+1, i+row_sensors, i+row_sensors+1]
            elif i%row_sensors == (row_sensors-1): edge['lst_{}'.format(i)]=[i-row_sensors, i-row_sensors-1, i-1, i+row_sensors-1, i+row_sensors]
            else: edge['lst_{}'.format(i)]=[i-row_sensors-1, i-row_sensors, i-row_sensors+1, i-1, i+1, i+row_sensors-1, i+row_sensors, i+row_sensors+1]

        for i in range(row_sensors):
            if i==0: edge['lst_{}'.format(i)]=[1,row_sensors,row_sensors+1,self.num_sensor-row_sensors,self.num_sensor-row_sensors+1]
            elif i==(row_sensors-1): edge['lst_{}'.format(i)]=[row_sensors-2,row_sensors*2-2,row_sensors*2-1,self.num_sensor-2,self.num_sensor-1]
            else: edge['lst_{}'.format(i)]=[i-1, i+1, i+row_sensors-1, i+row_sensors, i+row_sensors+1, i+self.num_sensor-row_sensors-1, 
                                            i+self.num_sensor-row_sensors, i+self.num_sensor-row_sensors+1]

        for i in range(int(self.num_sensor-row_sensors),self.num_sensor):
            if i==int(self.num_sensor-row_sensors): edge['lst_{}'.format(i)]=[i-row_sensors,i-row_sensors+1,i+1,0,1]
            elif i==(self.num_sensor-1): edge['lst_{}'.format(i)]=[i-row_sensors-1,i-row_sensors,i-1,row_sensors-2,row_sensors-1]
            else: edge['lst_{}'.format(i)]=[i-1, i-row_sensors-1, i-row_sensors, i-row_sensors+1, i+1, i-(self.num_sensor-row_sensors+1), 
                                            i-(self.num_sensor-row_sensors), i-(self.num_sensor-row_sensors-1)]        

        for i in range(row_sensors):
            if i%row_sensors==0: 
                edge['lst_{}'.format(i)]=[2,row_sensors+2,row_sensors*2,row_sensors*2+1,row_sensors*2+2,self.num_sensor-row_sensors+2,
                                          self.num_sensor-(row_sensors*2),self.num_sensor-(row_sensors*2)+1,self.num_sensor-(row_sensors*2)+2]
            
            elif i%row_sensors==1: 
                edge['lst_{}'.format(i)]=[3,row_sensors+3,row_sensors*2,row_sensors*2+1,row_sensors*2+2,row_sensors*2+3,self.num_sensor-5,self.num_sensor-(2*row_sensors),self.num_sensor-(2*row_sensors)+1,
                            self.num_sensor-(2*row_sensors)+2,self.num_sensor-(2*row_sensors)+3]
            elif i%row_sensors==(row_sensors-2): edge['lst_{}'.format(i)]=[4,row_sensors*2-4,row_sensors*2+4,row_sensors*3-3,row_sensors*3-2,row_sensors*3-1,self.num_sensor-4,self.num_sensor-12,
                                               self.num_sensor-11,self.num_sensor-10,self.num_sensor-9]
            elif i%row_sensors==(row_sensors-1):edge['lst_{}'.format(i)]=[5,row_sensors*2-3,row_sensors*3-3,row_sensors*3-2,
                                                            row_sensors*3-1,self.num_sensor-3,self.num_sensor-11,self.num_sensor-10,self.num_sensor-9]

        for i in range(row_sensors,int(row_sensors*2)):
            if i%row_sensors==0: edge['lst_{}'.format(i)]=[2,row_sensors+2,row_sensors*2,row_sensors*3,row_sensors*3+1,row_sensors*3+2,self.num_sensor-row_sensors,self.num_sensor-row_sensors+1,self.num_sensor-row_sensors+2]
            elif i%row_sensors==1: edge['lst_{}'.format(i)]=[3,row_sensors+3,row_sensors*2+3,row_sensors*3,row_sensors*3+1,row_sensors*3+2,row_sensors*3+3,self.num_sensor-row_sensors,self.num_sensor-row_sensors+1,
                                               self.num_sensor-row_sensors+2,self.num_sensor-row_sensors+3]
            elif i%row_sensors==(row_sensors-2): edge['lst_{}'.format(i)]=[4,row_sensors+4,row_sensors*2+4,row_sensors*3+4,row_sensors*4-3,row_sensors*4-2,row_sensors*4-1,self.num_sensor-4,self.num_sensor-3,self.num_sensor-2,self.num_sensor-1]
            elif i%row_sensors==(row_sensors-1): edge['lst_{}'.format(i)]=[5,row_sensors*2-3,row_sensors*3-3,row_sensors*4-3,row_sensors*4-2,row_sensors*4-1,self.num_sensor-3,self.num_sensor-2,self.num_sensor-1]

        for i in range(int(row_sensors*2),int(self.num_sensor-int(row_sensors*2))):
            if i%row_sensors==0:   edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors, i-row_sensors-row_sensors+1, i-row_sensors-row_sensors+2, i-row_sensors+2, i+1+1, 
                                             i+row_sensors+2, i+row_sensors+row_sensors, i+row_sensors+row_sensors+1, i+row_sensors+row_sensors+2]
                
            elif i%row_sensors==1: edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors-1, i-row_sensors-row_sensors, i-row_sensors-row_sensors+1, i-row_sensors-row_sensors+2, 
                                      i-row_sensors+2, i+1+1, i+row_sensors+2, i+row_sensors+row_sensors-1, i+row_sensors+row_sensors, i+row_sensors+row_sensors+1, i+row_sensors+row_sensors+2]
                
            elif i%row_sensors==(row_sensors-2): edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors-2, i-row_sensors-row_sensors-1, i-row_sensors-row_sensors+1, i-row_sensors-row_sensors+2, i-row_sensors-2, 
                                                             i-1-1, i+row_sensors-2, i+row_sensors+row_sensors-2, i+row_sensors+row_sensors-1, i+row_sensors+row_sensors, i+row_sensors+row_sensors+1]
                
            elif i%row_sensors==(row_sensors-1): edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors-2, i-row_sensors-row_sensors-1, i-row_sensors-row_sensors, i-row_sensors-2, i-1-1, i+row_sensors-2, 
                                                             i+row_sensors+row_sensors-2, i+row_sensors+row_sensors-1, i+row_sensors+row_sensors]
                
            else: edge['lst_{}'.format(i)]=[i-row_sensors-row_sensors-2, i-row_sensors-row_sensors-1, i-row_sensors-row_sensors, i-row_sensors-row_sensors+1, i-row_sensors-row_sensors+2, i-row_sensors-2, i-row_sensors+2, 
                              i-1-1, i+1+1, i+row_sensors-2, i+row_sensors+2, i+row_sensors+row_sensors-2, i+row_sensors+row_sensors-1, i+row_sensors+row_sensors, i+row_sensors+row_sensors+1, i+row_sensors+row_sensors+2]
                
        for i in range(int(self.num_sensor-(2*row_sensors)),int(self.num_sensor-row_sensors)):
            if i%row_sensors==0: edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*4),self.num_sensor-(row_sensors*4)+1,self.num_sensor-(row_sensors*4)+2,
                                             self.num_sensor-(row_sensors*2)-6,self.num_sensor-row_sensors-6,self.num_sensor-6,0,1,2]
            elif i%row_sensors==1: edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*4),self.num_sensor-(row_sensors*4)+1,self.num_sensor-(row_sensors*4)+2,self.num_sensor-(row_sensors*4)+3, 
                                               self.num_sensor-(row_sensors*4)+4,self.num_sensor-(row_sensors*3-3),self.num_sensor-row_sensors-5,self.num_sensor-5,0,1,2,3]
        
            elif i%row_sensors==(row_sensors-2): edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3+4),self.num_sensor-(row_sensors*3+3),self.num_sensor-(row_sensors*3+2),self.num_sensor-(row_sensors*3+1),self.num_sensor-(row_sensors*2+4),self.num_sensor-(row_sensors+4), row_sensors-4, row_sensors-3, row_sensors-2, row_sensors-1]
            elif i%row_sensors==(row_sensors-1): edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3+3),self.num_sensor-(row_sensors*3+2),self.num_sensor-(row_sensors*3+1),self.num_sensor-(row_sensors*2+3), self.num_sensor-(row_sensors+3),self.num_sensor-3,row_sensors-3,row_sensors-2,row_sensors-1]

        for i in range(int(self.num_sensor-row_sensors),self.num_sensor):
            if i%row_sensors==0: edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3),self.num_sensor-(row_sensors*3)+1,self.num_sensor-(row_sensors*3)+2,self.num_sensor-(row_sensors+6),self.num_sensor-6,2,row_sensors,row_sensors+1,row_sensors+2]
            elif i%row_sensors==1: edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3),self.num_sensor-(row_sensors*3)+1,self.num_sensor-(row_sensors*3)+2,self.num_sensor-(row_sensors*3)+3,
                                               self.num_sensor-row_sensors-5,self.num_sensor-5,3,row_sensors,row_sensors+1,row_sensors+2,row_sensors+3]
            elif i%row_sensors==(row_sensors-2): edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3)+4,self.num_sensor-(row_sensors*3)+5,self.num_sensor-(row_sensors*3)+6,self.num_sensor-(row_sensors*3)+7,
                                               self.num_sensor-(row_sensors+4),4,row_sensors+4,row_sensors+5,row_sensors+6,row_sensors+7]
            elif i%row_sensors==(row_sensors-1): edge['lst_{}'.format(i)]=[self.num_sensor-(row_sensors*3)+5,self.num_sensor-(row_sensors*3)+6,self.num_sensor-(row_sensors*3)+7,
                                               self.num_sensor-(row_sensors+5),row_sensors-3,5,row_sensors+5,row_sensors+6,row_sensors+7]
                
        src_tmp,tgt_tmp = [],[]
        num_cnt = 0

        for sensors in range(self.num_sensor):
            for i in range(len(edge['lst_{}'.format(sensors)])):
                src_tmp.append(sensors)
                tgt_tmp.append(edge['lst_{}'.format(sensors)][i])

        src, tgt = [], []

        src = src + src_tmp
        tgt = tgt + tgt_tmp

        for i in range(int(len(self.df)/self.num_sensor)-1):
            for idx in range(len(src_tmp)):
                src_tmp[idx] = src_tmp[idx] + self.num_sensor
                tgt_tmp[idx] = tgt_tmp[idx] + self.num_sensor

            src = src + src_tmp
            tgt = tgt + tgt_tmp
                
            for k in range(self.num_sensor):  
                tgt.append((i+1)*self.num_sensor + k)  
                src.append(num_cnt)
                num_cnt = num_cnt + 1

        return self.generate_graph(src, tgt)
    
    def draw_graph(self, input_Graph):
        plt.figure(figsize=(6, 6))
        #colors = graph1["label"].tolist()
        cora_graph = nx.from_pandas_edgelist(input_Graph)
        #subjects = list(graph1[graph1["label"].isin(list(cora_graph.nodes))]["label"])
        nx.draw_spring(cora_graph, node_size=15) #color=blue

