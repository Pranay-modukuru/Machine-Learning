# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:02:45 2019

@author: prana
"""

#%% Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from model import LinearRegression

#%% Setting random seed
np.random.seed(10)

#%%
if __name__ == '__main__':
        
    #%% Generate Data
    # Generate Random data
    x = np.random.randn(100,1)
    
    # Generate random noise [same shape as input]
    noise = np.random.randn(100,1)
    
    # Generate true outputs
    # ( y = mx + c)
    y_true = 3 * x + 2 + 0.2 * noise 
    
    #%% Training and Validation split
    
    shuffle_idx = np.random.permutation(x.size)
    
    x_train, y_train = x[shuffle_idx[:80]], y_true[shuffle_idx[:80]]
    x_valid, y_valid = x[shuffle_idx[80:]], y_true[shuffle_idx[80:]] 
    
    #%% Initiate model
    slope = np.random.rand(1,1)
    intercept = np.random.rand(1,1)
    
    model = LinearRegression(slope= slope,intercept= intercept, lr= 0.01)
    
    
    #%%
    print('\nBefore Training ')
    print('Slope : ', model.slope)
    print('Intercept : ', model.intercept)
    print('\n')
    
    
    #%% Start training
    
    losses = np.array([])
    
    model.set_mode('train')
    
    for i in range(1000):
        
        y_pred = model.predict(x_train)
        
        loss = model.MSEloss(y_pred, y_train)
        losses = np.append(losses, loss)
            
        model.update()
        
        if i % 100 == 0:
            print('Epoch : {:d}, Loss : {:.4f}'.format(i,loss))
        
    #%%
    print('\nAfter Training ')
    print('Slope : ', model.slope)
    print('Intercept : ', model.intercept)
    
    #%% Test performance of model
    
    model.set_mode('eval')
    
    y = model.predict(x_valid)
    loss = model.MSEloss(y, y_valid)
    
    print('\nLoss on Validation set : {:.4f}'.format(loss))
    #%% Plots
    
    fig = plt.figure(figsize = (15,6))
    
    # Loss
    ax = fig.add_subplot(121)
    ax.plot(range(losses.size),losses, label = 'training loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc = 'best')
    
    # Regression Line
    ax2 = fig.add_subplot(122)
    ax2.scatter(y_train, x_train, c = 'blue', label = 'training data')
    ax2.scatter(y_valid, x_valid, c ='orange', label = 'validation data')
    ax2.plot(y,x_valid, c = 'red',label = 'regression line')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(loc = 'best')
    