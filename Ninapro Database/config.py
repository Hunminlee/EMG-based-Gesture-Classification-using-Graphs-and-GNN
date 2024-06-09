#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf

class Graph_Ninapro:
    def __init__(self, df, num_sensor):
        self.df = df
        self.num_sensor = num_sensor
        self.row_sensor = 16
    
    def generate_graph(self, src, dst):
        graph = pd.DataFrame([src, dst]).T
        graph.columns = ['source', 'target']

        edges = graph[["source", "target"]].to_numpy().T
        edge_weights = tf.ones(shape=edges.shape[1])

        return graph, edges, edge_weights


    def KNN1(self):
        
        cnt = 0
        src,dst=[],[]
        for i in range(len(self.df)):
            cnt+=1
            
            if cnt == self.row_sensor:
                cnt=0
                
            if (i + 1) % int(self.row_sensor/2) == 0: 
                src.append(i)
                dst.append(i-int(self.row_sensor/2)+1)
                if (i+1)%self.row_sensor != 0:
                    src.append(i)
                    dst.append(i+int(self.row_sensor/2))
                
            else: 
                src.append(i)
                dst.append(i + 1)
                
                if cnt < int(self.row_sensor/2)+1:
                    dst.append(i+int(self.row_sensor/2))
                    src.append(i)
            
        return self.generate_graph(src, dst)
    

    def KNN1_SW(self):
        
        src,dst=[],[]
        for i in range(len(self.df)):
            src.extend([i,i])
            if (i+1)%8 == 0: 
                dst.append(i-int(self.row_sensor/2)+1)
                dst.append(i+int(self.row_sensor/2))
            else: 
                dst.append(i+1)
                dst.append(i+int(self.row_sensor/2))
            
        return self.generate_graph(src, dst)
    
    
    def KNN2(self):
        src = []
        num, dst = 0, []
        for i in range(len(self.df)): 
            src.extend([i, i, i])
            
        for i in range(0, len(self.df), int(self.row_sensor/2)):            
            for j in range(int(self.row_sensor/2)):
                if (num+1)%int(self.row_sensor/2)==0: dst.append(num-int(self.row_sensor/2)+1)
                else: dst.append(num+1)

                if j < 6: dst.append(i+j+2)
                elif j > 5: dst.append(i+j-6)
                    
                if j<2: dst.append(i+j+6)    
                elif j>1: dst.append(i+j-2)    
                
                num=num+1

        cnt=0
        for i in range(0, len(self.df), int(self.row_sensor/2)):
            cnt+=1
            if cnt%2==1:       
                for j in range(int(self.row_sensor/2)):
                    src.append(i+j)
                    dst.append(i+j+int(self.row_sensor/2))
            
        return self.generate_graph(src, dst)


    def KNN2_SW(self):
        src=[]
        for i in range(len(self.df)): 
            src.extend([i, i, i, i])
            
        num, dst= 0, []
        for i in range(0, len(self.df), int(self.row_sensor/2)):
            for j in range(int(self.row_sensor/2)):
                if (num+1)%int(self.row_sensor/2)==0: dst.append(num-int(self.row_sensor/2)+1)
                else: dst.append(num+1)
                
                if j < 6: dst.append(i+j+2)
                elif j > 5: dst.append(i+j-6)
                    
                if j<2: dst.append(i+j+6)    
                elif j>1: dst.append(i+j-2)    

                dst.append(i+j+int(self.row_sensor/2))
                num=num+1

        return self.generate_graph(src, dst)

    def FC(self):
        src, dst = [], []
        
        for i in range(len(self.df)):
            src.extend([i] * (self.row_sensor-1))
                
        for i in range(0, len(self.df), self.row_sensor):
            for j in range(self.row_sensor):
                for k in range(self.row_sensor):
                    if j!=k: dst.append(i+k)
                
        for i in range(len(self.df)):
            if i % self.row_sensor <= int(self.row_sensor/2):
                src.extend([i, i, i])
                dst.extend([i+int(self.row_sensor/2)-1, i+int(self.row_sensor/2), i+int(self.row_sensor/2)+1])
            
        return self.generate_graph(src, dst)
    
    def FC(self):
        src, dst = [], []
        
        for i in range(len(self.df)): 
            src.extend([i] * (self.row_sensor-1))
                
        for i in range(0, len(self.df), self.row_sensor):
            for j in range(self.row_sensor):
                for k in range(self.row_sensor):
                    if j!=k: dst.append(i+k)
                
        for i in range(len(self.df)):
            if i < len(self.df)-self.row_sensor:
                src.append(i)
                dst.append(i+self.row_sensor)

            if i % self.row_sensor <= int(self.row_sensor/2):
                src.extend([i, i, i])
                dst.extend([i+int(self.row_sensor/2)-1, i+int(self.row_sensor/2), i+int(self.row_sensor/2)+1])
            
        return self.generate_graph(src, dst)

    def draw_graph(self, input_Graph):
        plt.figure(figsize=(6, 6))
        #colors = graph1["Label"].tolist()
        cora_graph = nx.from_pandas_edgelist(input_Graph)
        #subjects = list(graph1[graph1["Label"].isin(list(cora_graph.nodes))]["Label"])
        nx.draw_spring(cora_graph, node_size=15) #color=blue

