#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:31:16 2025

@author: matti
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.linalg import inv


# Define LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(LSTMAutoencoder, self).__init__()
        
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.latent = nn.Linear(hidden_size, latent_size)
        
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape  # Get original sequence length
    
        _, (h, _) = self.encoder(x)  # h has shape (1, batch, hidden_size)
        z = self.latent(h[-1])  # Shape: (batch, latent_size)
    
        # Repeat latent vector for each time step
        z = z.unsqueeze(1).expand(-1, seq_len, -1)  # Shape: (batch, seq_len, latent_size)
    
        h_dec = self.decoder_input(z)  # Convert latent to hidden size
        out, _ = self.decoder(h_dec)  # Decode entire sequence
    
        return out  # Output shape should be (batch, seq_len, input_size)

#define Fully connected autoencoder
class FullyConectedAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(FullyConectedAutoencoder, self).__init__()
        
        middle_dim = 8
        self.encoder = nn.Sequential(
            nn.Linear(input_size,middle_dim),
            nn.Tanh(),
            nn.Linear(middle_dim, latent_size),
            nn.Tanh(),
            )
        
        self.decoder= nn.Sequential(
            nn.Linear(latent_size,middle_dim),
            nn.Tanh(),
            nn.Linear(middle_dim,input_size),
            #nn.Tanh()
            )
        
    def forward(self,X):
        
        latent = self.encoder(X)
        out = self.decoder(latent)
        
        return out
    
class GPDenoiser():
    def __init__(self,input_dim,obs_noise = 0.1, coherence_length = 0.1):
        self.input_dim = input_dim
        self.obs_noise = obs_noise
        self.coherence_length = coherence_length
        self.domain = np.linspace(0, 1,input_dim)
        x1 = np.array([self.domain] * input_dim)
        x2 = np.array([self.domain] * input_dim).T
     
        self.kernel = np.exp(-1 * np.abs(x1-x2)/self.coherence_length)
        
        self.transform_matrix = np.dot(self.kernel,inv(self.kernel + obs_noise*np.identity(input_dim)))
    
    def forward(self,X):
        mean = np.mean(X)
        X -= mean
        return (self.transform_matrix @ X) + mean
       
   
    












        
        