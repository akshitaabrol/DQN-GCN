# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:47:25 2024

@author: Akshita
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Flatten
from stellargraph.layer.gcn import GraphConvolution
from stellargraph.layer.sort_pooling import SortPooling
import numpy as np
import argparse


def create_mapping(source_range, dest_range, n):
    mapping = {}
    action_sequence = range(n)

    for source in range(source_range):
        for dest in range(dest_range):
            for action in action_sequence:
                if source != dest:
                   mapping[(source+1, dest+1, action)] = len(mapping)

    print (mapping)
    return mapping
#can use this to preprocess your data
def process_data(state, node_feature, demand, source, destination):
    src_req = [1 if x == source else 0 for x in range(len(node_feature))]
    dest_req = [1 if x == destination else 0 for x in range(len(node_feature))]
    n_req = np.vstack((np.array(src_req),np.array(dest_req)))
    n_req = np.vstack((n_req, np.full_like(src_req, demand/100, dtype=float)))
    n_req = np.transpose(n_req)
    net_state = np.column_stack((node_feature,state,n_req))
    return net_state

# Define the model class
class ActionStateModel:
    def __init__(self, state_dim, action_dim, feature_size, learning_rate=0.00025):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_size = feature_size
        
        self.model = self.create_model(learning_rate)
    
    def create_model(self, learning_rate):
        x_features = Input(shape=(self.state_dim, self.feature_size))
        mask = Input(shape=(self.state_dim,), dtype=bool)
        x_adjacency = Input(shape=(self.state_dim, self.state_dim))
        x_inp = [x_features, x_adjacency]
        
        x = GraphConvolution(128, activation='relu')([x_features, x_adjacency])
        x = GraphConvolution(128, activation='relu')([x, x_adjacency])
        x = GraphConvolution(128, activation='relu')([x, x_adjacency])
        x = GraphConvolution(64, activation='relu')([x, x_adjacency])
        
        x = SortPooling(k=14, flatten_output=True)(x, mask)
        x = Conv1D(filters=32, kernel_size=64, strides=64)(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=5, strides=1)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=[x_features, x_adjacency, mask], outputs=output)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate))
        return model
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def predict(self, state):
        return self.model.predict(state)

# Define the agent class
class Agent:
    def __init__(self, state_dim, action_dim, feature_size, adj_mat, weights_path):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_size = feature_size
        self.adj_mat = adj_mat
        self.source_mapping = create_mapping(self.state_dim, self.state_dim, 3)
        self.model = ActionStateModel(state_dim, action_dim, feature_size)
        self.model.load_weights(weights_path)
    
    def get_action(self, state, source, destination):
        
        mask = np.ones((1, self.state_dim), dtype=bool)
        q_value = self.model.predict([np.expand_dims(state, 0), np.expand_dims(self.adj_mat,0) , mask])[0]        
        actions = [self.source_mapping.get((source, destination, x), -1) for x in range(3)]        
        return actions[np.argmax(q_value[actions])]

   
#change this according to your implementation of sending the features
    def test(self):
                #dummy data
            while(1):
                
                #needs to be provided by controller
                node_feature = np.random.rand(5, 2) #node degree normalized by dividing by max degree and node imporatnce 
                state = np.random.rand(5, 2) #tx and rx of every node normalized by dividing with max link capacity
                source = 1; #need to convert it to one-hot encoder 
                destination = 2;  #need to convert it to one-hot encoder
                demand = 20; #need to normalize it by dividing by max link capacity
                
                obs = process_data(state, node_feature, demand, source, destination)
                
                action = self.get_action(obs,source,destination)
                print(f'Source{source},Destination{destination}')
                print(f'Action: {action}')
                
                
                delay = 0.01; #needs to be returned from controller
                throughput = 20; #needs to be returned from controller
                

# Main function to run the agent
def main():
    dir_path = "%s/weights"%os.getcwd()
    weights_path = dir_path+'/'+'gym_dqn_model_weights.h5'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=weights_path, help="Path to the trained model weights")    
    
    args = parser.parse_args()

    adj_mat = np.load("adjacency_matrix.npy")
    print(f'Action: {adj_mat}')
    state_dim = 5
    action_dim = 60
    feature_size=7

    agent = Agent(state_dim, action_dim, feature_size, adj_mat, args.weights)
    agent.test()

if __name__ == "__main__":
    main()
