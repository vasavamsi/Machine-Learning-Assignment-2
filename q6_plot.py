import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))
#y = pd.Series(y)
"""
X = PolynomialFeatures(degree = 2)
data_set = pd.DataFrame(X.transform(x))
data_set['label'] = y
data_set = data_set.sample(n = 5).reset_index(drop = True)
print(data_set) 
y = pd.Series(data_set['label'])
print(y)
"""
X = PolynomialFeatures(degree = deg)
data = pd.DataFrame(X.transform(x))
data['label'] = y
degrees = [1, 3, 5, 7]

for N in range(1,13):  ## Taking the no. of samples randomly
    max_theta_val = []
    for deg in degrees: ##Taking the degrees into account  
        data_set = data.sample(n = N*5).reset_index(drop = True)
        y = pd.Series(data_set['label'])
        LR = LinearRegression(fit_intercept=True)
        theta = LR.fit_vectorised(data_set, y,1,n_iter=1000)
        
        max_theta_val.append(max(abs(theta)))
    plt.figure(N)
    plt.scatter(degrees, max_theta_val)
