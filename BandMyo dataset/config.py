#!/usr/bin/env python
# coding: utf-8

# config.py: This file contains configuration settings for your program. It might include variables that control various aspects of the program's behavior, such as file paths, API keys, database connection details, etc.


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf

class Graph_BandMyo:
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
        K=1
        src = [i for i in range(len(self.df))]
        dst = [(i-(self.num_sensor-K)) if (i+K) % self.num_sensor == 0 else (i+K) for i in range(len(self.df))]
        #dst=[]
        #for i in range(len(self.df)):
        #    if (i+1)%self.num_sensor == 0: dst.append(i-(self.num_sensor-1))
        #    else: dst.append(i+1)

        return self.generate_graph(src, dst)

    def KNN1_SW(self):
        src, dst = [], []
        K=1
        for i in range(len(self.df)):
            src.append(i)
            if i<len(self.df):
                src.append(i)
                dst.append(i+self.num_sensor)
            if (i+K)%self.num_sensor == 0: dst.append(i-(self.num_sensor-K))
            else: dst.append(i+K)

        return self.generate_graph(src, dst)

    def KNN2(self):
        K=2
        src, dst, num = [i for i in range(len(self.df)) for _ in range(K+1)], [], 0
        for i in range(0, len(self.df), self.num_sensor):
            for j in range(self.num_sensor):
                if (num+1)%self.num_sensor==0: dst.append(num-(self.num_sensor-1))
                else: dst.append(num+1)
                
                if j<(self.num_sensor-K): dst.append(i + j+K)
                elif j>(self.num_sensor-K-1): dst.append(i + j-(self.num_sensor-K))
                    
                if j<K: dst.append(i + j+(self.num_sensor-K))    
                elif j>(K-1): dst.append(i + j-K)    
                
                num+=1
                
        #src.extend([i for i in range(self.num_sensor)])
        #dst.extend([i+self.num_sensor for i in range(self.num_sensor)])
    
        return self.generate_graph(src, dst)

    
    def KNN2_SW(self):
        K=2
        src, dst, num = [i for i in range(len(self.df)) for _ in range(K+1)], [], 0

        for i in range(0, len(self.df), 8):
            for j in range(8):
                if (num+1)%8==0: dst.append(num-(8-1))
                else: dst.append(num+1)

                if j<(8-K): dst.append(i + j+K)
                elif j>(8-K-1): dst.append(i + j-(8-K))

                if j<K: dst.append(i + j+(8-K))    
                elif j>(K-1): dst.append(i + j-K)    

                num+=1
        
        src_dst_pairs = [(i, i + 8) for i in range(len(self.df)) if i < len(self.df) - 8]
        src.extend([pair[0] for pair in src_dst_pairs])
        dst.extend([pair[1] for pair in src_dst_pairs])
        

        return self.generate_graph(src, dst)

    
    def FC(self):
        num, src, dst = self.num_sensor, [], []

        for i in range(len(self.df)):
            if i % self.num_sensor == 0 and i > 0:
                num += self.num_sensor
            dst.extend(range(i + 1, num))
            src.extend([i] * (num - i - 1))

        return self.generate_graph(src, dst)

    
    def FC_SW(self):
        num, src, dst = self.num_sensor, [], []

        for i in range(len(self.df)):
            if i < len(self.df)-self.num_sensor:
                src.append(i)
                dst.append(i+self.num_sensor)
                
            if i % self.num_sensor == 0 and i > 0:
                num += self.num_sensor
            dst.extend(range(i + 1, num))
            src.extend([i] * (num - i - 1))

        return self.generate_graph(src, dst)
    

    
    def draw_graph(self, input_Graph):
        plt.figure(figsize=(6, 6))
        #colors = graph1["Label"].tolist()
        cora_graph = nx.from_pandas_edgelist(input_Graph)
        #s_idxs = list(graph1[graph1["Label"].isin(list(cora_graph.nodes))]["Label"])
        nx.draw_spring(cora_graph, node_size=15)






