# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:17:07 2020

@author: ADMIN-PC
"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import autograd.numpy as np
# Import Autograd modules here
from autograd import grad


class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.mse_list = []
        self.t_0 = []
        self.t_1 = []
        
        pass

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        data = X
        label = y
        ##Running iterations
        for itr in range(n_iter):
            if batch_size == 1 and itr == 0:
                X = data.values
                y = label.values
                row, col = np.shape(X)
                
                ##Taing care of intercept
                if self.fit_intercept == True:
                    intercept = np.ones((row,col+1))
                    intercept[:, 1:(col+1)] = X
                    fin_X = intercept
                else:
                    fin_X = X
                row,col = np.shape(fin_X)
            elif batch_size != 1:
                data['label'] = label
                X = data.sample(batch_size)
                X = X.values
                y = X[:,-1]
                X = X[:,0:X.shape[1]-1]
                row, col = np.shape(X)
                
                ##Taing care of intercept
                if self.fit_intercept == True:
                    intercept = np.ones((row,col+1))
                    intercept[:, 1:(col+1)] = X
                    fin_X = intercept
                else:
                    fin_X = X
                row,col = np.shape(fin_X)
            #print(fin_X.shape)
             
            ##Initiating the coefs
            if itr == 0:
                self.coef_ = np.random.uniform(size=col) ##Taking the initial guess for gradient descent
            
            ##Taking care of learning rate
            if lr_type != 'constant':
                lr =lr/t
            for sample in range(0,fin_X.shape[0]):
                ##Updating the coefs
                fin_X_sample = fin_X[sample,:]
                for i in range(col):  ##col+1
                    y_hat = np.dot(fin_X_sample,self.coef_)
                    err = y - y_hat
                    temp = np.dot(err, fin_X[:,i]*-1)
                    diff = float(2*(temp))/float(row)
                    
                    ##Updating the coefficient values
                    self.coef_[i] = self.coef_[i] - lr*diff
        print(self.coef_.shape)
        
        #pass

    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.0000000000001, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        data = X
        label = y
        ##Running iterations
        for itr in range(n_iter):
            if batch_size == 1 and itr == 0:
                X = data.values
                y = label.values
                row, col = np.shape(X)
                
                ##Taing care of intercept
                if self.fit_intercept == True:
                    intercept = np.ones((row,col+1))
                    intercept[:, 1:(col+1)] = X
                    fin_X = intercept
                else:
                    fin_X = X
                row,col = np.shape(fin_X)
            elif batch_size != 1:
                data['label'] = label
                X = data.sample(batch_size)
                X = X.values
                y = X[:,-1]
                X = X[:,0:X.shape[1]-1]
                row, col = np.shape(X)
                
                ##Taing care of intercept
                if self.fit_intercept == True:
                    intercept = np.ones((row,col+1))
                    intercept[:, 1:(col+1)] = X
                    fin_X = intercept
                else:
                    fin_X = X
                row,col = np.shape(fin_X)
            #print(fin_X.shape)
             
            ##Initiating the coefs
            if itr == 0:
                self.coef_ = np.random.uniform(size=col) ##Taking the initial guess for gradient descent
            
            ##Taking care of learning rate
            if lr_type != 'constant':
                lr =lr/t
            diff_list = []
            
            ##Generating the gradient vector
            for i in range(col):  ##col+1
                y_hat = np.dot(fin_X,self.coef_)
                err = y - y_hat
                temp = np.dot(err, fin_X[:,i]*-1)
                diff = float(2*(temp))/float(row)
                diff_list.append(diff)
            grad_vector = np.array(diff_list)
            
            ##Updating all the co-efficients at once
            self.coef_ = self.coef_ - lr*grad_vector
        
        return self.coef_
        #pass

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        def MSE(coef_):
            y_hat = np.dot(fin_X,coef_)
            
            mse = sum((y-y_hat)**2)/row
            
            return mse
        data = X
        label = y
        ##Running iterations
        for itr in range(n_iter):
            if batch_size == 1 and itr == 0:
                X = data.values
                y = label.values
                row, col = np.shape(X)
                
                ##Taing care of intercept
                if self.fit_intercept == True:
                    intercept = np.ones((row,col+1))
                    intercept[:, 1:(col+1)] = X
                    fin_X = intercept
                else:
                    fin_X = X
                row,col = np.shape(fin_X)
            elif batch_size != 1:
                data['label'] = label
                X = data.sample(batch_size)
                X = X.values
                y = X[:,-1]
                X = X[:,0:X.shape[1]-1]
                row, col = np.shape(X)
                
                ##Taing care of intercept
                if self.fit_intercept == True:
                    intercept = np.ones((row,col+1))
                    intercept[:, 1:(col+1)] = X
                    fin_X = intercept
                else:
                    fin_X = X
                row,col = np.shape(fin_X)
            #print(fin_X.shape)
             
            ##Initiating the coefs
            if itr == 0:
                self.coef_ = np.random.uniform(size=col) ##Taking the initial guess for gradient descent
            
            ##Taking care of learning rate
            if lr_type != 'constant':
                lr =lr/t
            self.mse_list.append(MSE(self.coef_))
            training_grad_fun = grad(MSE)
            diff = training_grad_fun(self.coef_) ##Finding the gradients using the autograd module
            
            self.coef_ -= lr*diff ##Updating the co-efficients
            self.t_0.append(self.coef_[0])
            self.t_1.append(self.coef_[1])
        
        return self.t_0, self.t_1
    
    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        X = X.values
        y = y.values
        self.y = y
        row,col = np.shape(X) # taking into account the shape of X
        
        #considering the intercept option
        if self.fit_intercept == True:
            intercept = np.ones((row,col+1))
            intercept[:, 1:(col+1)] = X
            fin_X = intercept
        else:
            fin_X = X
        
        #Applying the normal equation
        X_trans = np.transpose(fin_X)
        product_1 = X_trans.dot(fin_X)
        product_1_inv = np.linalg.inv(product_1)
        product_2 = product_1_inv.dot(X_trans)
        self.coef_ = product_2.dot(y)
        pass

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X = X.values
        row,col = np.shape(X)
        ##Taing care of intercept
        if self.fit_intercept == True:
            intercept = np.ones((row,col+1))
            intercept[:, 1:(col+1)] = X
            fin_X = intercept
        else:
            fin_X = X
        y_hat = np.dot(fin_X, self.coef_)
        return pd.Series(y_hat)
        #pass

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        x1 = np.linspace(0, 10.0, 10)
        x2 = np.linspace(0, 10.0, 10)
        X1, X2 = np.meshgrid(x1, x2)
        X = X.values
        row,col = X.shape
        intercept = np.ones((row,col+1))
        intercept[:, 1:(col+1)] = X
        fin_X = intercept
        y = y.values
        T_0, T_1 = np.meshgrid(t_0[0:10], t_1[0:10])
        mse = np.zeros((10,10))
        cnt = 1
        print(T_0.shape)
        print(T_1.shape)
        print(mse.shape)

        for i in range(0,10):
            for j in range(0,10):
#                if i != j:
#                    continue
#                else:
                coef = np.array([T_0[i,j], T_1[i,j]])
                y_hat = np.dot(fin_X,coef)
                mse[i,j] = (sum((y-y_hat)**2))/y.shape[0]
                fig = plt.figure(cnt,figsize =(10,10))
                ax = fig.add_subplot(111, projection = '3d')
                ax.plot_surface(X1, X2, mse)
                plt.xlabel('t_0')
                plt.ylabel('t_1')
                plt.title('mse = {}'.format(mse[i,j]))
                plt.savefig('surface_{}.png'.format(cnt))
                cnt += 1
        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        X = X.values
        y = y.values
        x_axis = range(1,6)
        x_axis = np.array(x_axis)
        t_0 = np.array(t_0)
        t_1 = np.array(t_1)
        for i in range(0,10):
            y_axis = t_0[i] + x_axis*t_1[i]
            plt.plot(x_axis,y_axis)
            plt.scatter(X,y)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('t_0 = {} and t_1 = {}'.format(t_0[i],t_1[i]))
            plt.savefig("plot_{}.png".format(i+1))
        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        
        coefs = [t_0,t_1]
        coefs = np.array(coefs)
        coefs = coefs.transpose()
        """
        x1 = np.linspace(0, 10.0, 10)
        x2 = np.linspace(0, 10.0, 10)
        X1, X2 = np.meshgrid(x1, x2)
        X = X.values
        row,col = X.shape
        intercept = np.ones((row,col+1))
        intercept[:, 1:(col+1)] = X
        fin_X = intercept
        y = y.values
        T_0, T_1 = np.meshgrid(t_0[0:10], t_1[0:10])
        mse = np.zeros((10,10))
        cnt = 1
        for i in range(0,10):
            for j in range(0,10):
#                if i != j:
#                    continue
#                else:
                coef = np.array([T_0[i,j], T_1[i,j]])
                y_hat = np.dot(fin_X,coef)
                mse[i,j] = (sum((y-y_hat)**2))/y.shape[0]
        
                cp = plt.contour(X1, X2, mse, colors='black', linestyles='dashed', linewidths=1)
                plt.clabel(cp, inline=1, fontsize=10)
                plt.xlabel('t_0')
                plt.ylabel('t_1')
                plt.title('mse = {}'.format(mse[i,j]))
                plt.savefig('contour_{}.png'.format(cnt))
                plt.show()
                cnt += 1
        
        pass
