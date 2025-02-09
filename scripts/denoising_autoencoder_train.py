#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:50:05 2025

@author: matti
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from rl_models.denoising_autoencoder import FullyConectedAutoencoder
from torch.utils.data import DataLoader, TensorDataset


def generate_data_sin(dataset_length,input_size):
    domain = np.linspace(0,2,input_size)
    domain = np.array([domain] * dataset_length)
    phase = np.random.uniform(0,1,dataset_length)
    phase = np.array([phase] * input_size).T
    noise = 0.4 * np.random.randn(dataset_length,input_size)
    
    Y = np.sin(2 * np.pi * (domain + phase))
    X = Y + noise
    
    return X,Y

def generate_data_GP(dataset_length,input_size,coherence_length = 50,noise = 0.1):
    domain = np.linspace(0,1,input_size)
    x1 = np.array([domain] * input_size)
    x2 = np.array([domain] * input_size).T
    d_2 = -1*(x1 - x2)**2
    samples = []
    for j in range(0,dataset_length):
        l = np.random.uniform(0.005,0.1)
        # Sample from multivariate normal distribution
        y = np.random.multivariate_normal(mean=np.zeros(input_size), cov=np.exp(d_2/l))
        samples.append(y)
        
    noise_arr = noise * np.random.randn(dataset_length,input_size)    
    return np.array(samples)+ noise_arr, np.array(samples) 




input_size = 50
model = FullyConectedAutoencoder(input_size, 35, 30)


# Create TensorDataset and DataLoader
train_X, train_Y = generate_data_GP(10000, 50,noise = 0.3)
test_X, test_Y = generate_data_GP(2000, 50,noise = 0.3)
train_X, train_Y, = torch.tensor(train_X,dtype=torch.float32), torch.tensor(train_Y,dtype=torch.float32)
test_X, test_Y = torch.tensor(test_X,dtype=torch.float32), torch.tensor(test_Y,dtype=torch.float32)
train_dataset = TensorDataset(train_X,train_Y)
test_dataset = TensorDataset(test_X,test_Y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# Define MSE Loss
loss_func = nn.MSELoss()

# Define optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs and batch size
num_epochs = 50
batch_size = 64

# Assuming `train_loader` is already defined (from the previous code)
for epoch in range(num_epochs):
    train_loss = 0.0
    test_loss = 0.0
    
    #set model to train ()
    model.train()
    
    # Iterate over batches of data from the DataLoader
    for data, label in train_loader:
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass: Get the reconstructed output from the autoencoder
        output = model(data)
        
        # Calculate the loss: MSE between input and output
        loss = loss_func(output, label)
        
        # Backward pass: Compute gradients
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        # Track running loss (for reporting purposes)
        train_loss += loss.item()
    
    
    #evaluate
    model.eval()
    for data, label in test_loader:
        
        #do not calculate gradients
        with torch.no_grad():
            
            output = model(data)
            
            loss = loss_func(data,label)
            
            test_loss += loss.item()
            
        
        
    # Print loss every epoch
    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")
    
model.eval()

with torch.no_grad():
    total_loss = 0.0
    for data, _ in train_loader:
        output = model(data)
        loss = loss_func(output, data)
        total_loss += loss.item()
    
    avg_test_loss = total_loss / len(train_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
#%%
#gp = GPDenoiser(50,0.3,0.4)
sample_data , function = generate_data_GP(1,50,noise=0.2)
sample_data = torch.tensor(sample_data,dtype=torch.float32)
prediciton = model(sample_data)

sample_data = sample_data.squeeze(0).detach().numpy()
prediciton = prediciton.squeeze(0).detach().numpy()

plt.plot(sample_data,label = 'input')
#plt.plot(gp.forward(sample_data), label = 'GP')
plt.plot(prediciton, label = 'prediciton')
plt.plot(function[0], label = 'label')
plt.legend(loc=1)
#%%





