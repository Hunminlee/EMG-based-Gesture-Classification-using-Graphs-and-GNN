#!/usr/bin/env python
# coding: utf-8

# config.py: This file contains configuration settings for your program. It might include variables that control various aspects of the program's behavior, such as file paths, API keys, database connection details, etc.


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf

class Graph_CapgMyo:
    def __init__(self, df, num_sensor):
        self.df = df
        self.num_sensor = num_sensor
        self.row_sensor = 8
    
    def generate_graph(self, src, dst):
        graph = pd.DataFrame([src, dst]).T
        graph.columns = ['source', 'target']

        edges = graph[["source", "target"]].to_numpy().T
        edge_weights = tf.ones(shape=edges.shape[1])

        return graph, edges, edge_weights


    def KNN1(self):
        
        src,dst=[],[]
        for i in range(len(self.df)):
            src.append(i)
            if (i + 1) % self.row_sensor == 0: dst.append(i - (self.row_sensor-1))
            else: dst.append(i + 1)
            
        return self.generate_graph(src, dst)
    

    def KNN1_SW(self):
        
        src,dst=[],[]
        for i in range(len(self.df)):
            src.append(i)
            if (i + 1) % self.row_sensor == 0: dst.append(i - (self.row_sensor-1))
            else: dst.append(i + 1)
                
            if i < len(self.df) - self.row_sensor:
                src.append(i)
                dst.append(i + self.row_sensor)
            
        return self.generate_graph(src, dst)
    
    
    def KNN2(self):
        num, src, dst= 0, [], []

        for i in range(len(self.df)):
            src.extend([i, i, i])

        for i in range(0, len(self.df), self.row_sensor):
            for j in range(self.row_sensor):
                if (num+1)%self.row_sensor==0: dst.append(num-(self.row_sensor-1))
                else: dst.append(num+1)

                if j<(self.row_sensor-2): dst.append(i+j+2)
                elif j>(self.row_sensor-3): dst.append(i+j-(self.row_sensor-2))

                if j<2: dst.append(i+j+(self.row_sensor-2))
                elif j>1: dst.append(i+j-2)

                num=num+1
            
        return self.generate_graph(src, dst)


    def KNN2_SW(self):
        num, src, dst= 0, [], []

        for i in range(len(self.df)):
            src.extend([i, i, i])

        for i in range(0, len(self.df), self.row_sensor):
            for j in range(self.row_sensor):
                if (num+1)%self.row_sensor==0: dst.append(num-(self.row_sensor-1))
                else: dst.append(num+1)

                if j<(self.row_sensor-2): dst.append(i+j+2)
                elif j>(self.row_sensor-3): dst.append(i+j-(self.row_sensor-2))

                if j<2: dst.append(i+j+(self.row_sensor-2))
                elif j>1: dst.append(i+j-2)

                num=num+1
        
        for i in range(len(self.df)):
            if i < len(self.df)-self.num_sensor:
                src.append(i)
                dst.append(i+self.num_sensor)

        return self.generate_graph(src, dst)
    

    def FC(self):
        src, dst = [], []
        num=8

        for i in range(len(self.df)):
            if i%self.row_sensor==0 and i>0:
                num = num + self.row_sensor
            for j in range(i+1, num):
                dst.append(j)
                src.append(i)

        return self.generate_graph(src, dst)
    
    def FC_SW(self):
        src, dst = [], []
        num=8

        for i in range(len(self.df)):
            if i%self.row_sensor==0 and i>0:
                num = num + self.row_sensor
            for j in range(i+1, num):
                dst.append(j)
                src.append(i)
        
        for i in range(len(self.df)):
            if i < len(self.df)-self.num_sensor:
                src.append(i)
                dst.append(i+self.num_sensor)

        return self.generate_graph(src, dst)

    def draw_graph(self, input_Graph):
        plt.figure(figsize=(6, 6))
        #colors = graph1["Class"].tolist()
        cora_graph = nx.from_pandas_edgelist(input_Graph)
        #subjects = list(graph1[graph1["Class"].isin(list(cora_graph.nodes))]["Class"])
        nx.draw_spring(cora_graph, node_size=15) #color=blue

