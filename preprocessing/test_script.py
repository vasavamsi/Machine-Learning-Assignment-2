# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:59:24 2020

@author: ADMIN-PC
"""
import numpy as np

def transform(X,degree = 2):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        out = [1]
        for deg in range(1,degree+1):
            for num in X:
                out.append(num**deg)
        return np.array(out)

X = np.array([2,3, 4])
print(transform(X))