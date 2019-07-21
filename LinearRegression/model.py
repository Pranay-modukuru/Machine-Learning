#%% Importing Libraries
import numpy as np

#%% Build model

class LinearRegression():
    r"""
    Linear Regression class
    """
    def __init__(self, slope, intercept, lr):
        self.slope = slope
        self.intercept = intercept
        self.lr = lr
        self.train = True
        
    def predict(self, data):
        r""" 
        Predicts output given input data
        """
        out = np.dot(self.slope, data.T) + self.intercept
        self.d = data
        return out.T
    
    def MSEloss(self,predicted, true):
        r"""
        Calculates MSE loss and also calculates gradients of slope, intercept
        """
        error = predicted - true
        loss = (error**2).mean()
        
        if self.train == True:
            # Gradients calculated only during training
            self.slope_grad =  2 * (self.d * error).mean()
            self.intercept_grad = 2 * error.mean()
        
        return loss
    
    def set_mode(self, mode):
        if mode == 'train':
            self.train = True
        elif mode == 'eval':
            self.train = False
        
    def update(self):
        r"""
        Updates parameters and makes gradients zero after update
        """
        self.slope -= self.lr * self.slope_grad
        self.intercept -= self.lr * self.intercept_grad
        
        self.slope_grad = 0
        self.intercept_grad = 0


