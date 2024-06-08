#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense



class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.embedding_prepare = embedding_layers(hidden_units, dropout_rate)
        self.update_fn = embedding_layers(hidden_units, dropout_rate)
    
    def prepare(self, node_repesentations, weights=None):
        messages = self.embedding_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        if self.combination_type == "concat":
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        node_embeddings = self.update_fn(h)
        
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        node_repesentations, edges, edge_weights = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        
        aggregated_messages = self.aggregate(node_indices, neighbour_messages, node_repesentations)
        
        return self.update(node_repesentations, aggregated_messages)


def embedding_layers(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super(GNNNodeClassifier, self).__init__(*args, **kwargs)

        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        self.EncodingProcess = embedding_layers(hidden_units, dropout_rate, name="Encoding_layer")
        self.GraphConv1 = GraphConvLayer(hidden_units, dropout_rate,aggregation_type,combination_type,normalize,name="graph_conv1")
        self.GraphConv2 = GraphConvLayer(hidden_units,dropout_rate,aggregation_type,combination_type,normalize,name="graph_conv2")
        self.DecodingProcess = embedding_layers(hidden_units, dropout_rate, name="Decoding_layer")
        self.compute_logits = layers.Dense(units=num_classes, name="logits") 

    def call(self, input_node_indices):
        x = self.EncodingProcess(self.node_features) 
        x1 = self.GraphConv1((x, self.edges, self.edge_weights))         
        x = x1 + x 
        x2 = self.GraphConv2((x, self.edges, self.edge_weights)) 
        x = x2 + x 
        x = self.DecodingProcess(x) 
        node_embeddings = tf.gather(x, input_node_indices)  
        
        return self.compute_logits(node_embeddings)

def step_decay(epoch):
    #lr_schedule = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    lr_schedule = [0.01, 0.005, 0.005, 0.002, 0.001, 0.0005, 0.0001]
    #lr_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
    lr = lr_schedule[min(epoch // 20, len(lr_schedule) - 1)]
    return lr




'''def create_lstm(hidden_units, dropout_rate):
    inputs = keras.layers.Input(shape=(2, hidden_units[0]))
    x = inputs
    
    for units in hidden_units:
        x = layers.LSTM(
          units=units,
          activation="tanh",
          recurrent_activation="sigmoid",
          return_sequences=True,
          dropout=dropout_rate,
          return_state=False,
          recurrent_dropout=dropout_rate,
        )(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_GRU(hidden_units, dropout_rate):
    inputs = keras.layers.Input(shape=(2, hidden_units[0]))
    x = inputs
    
    for units in hidden_units:
        x = layers.GRU(
          units=units,
          activation="relu",
          recurrent_activation="sigmoid",
          return_sequences=True,
          dropout=dropout_rate,
          return_state=False,
          recurrent_dropout=dropout_rate,
        )(x)
    return keras.Model(inputs=inputs, outputs=x)
'''

