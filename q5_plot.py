import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
#from sklearn.linear_model import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))
y = pd.Series(y)
max_theta_val = []
deg_val = []
for deg in range(1,10): ##Taking limit upto 10 degrees
    X = PolynomialFeatures(degree = deg)
    data_set = pd.DataFrame(X.transform(x))
    LR = LinearRegression(fit_intercept=True)
    theta = LR.fit_vectorised(data_set, y,1,n_iter=1000)
    #theta = LR.fit(data_set, y)
    #fin_theta = theta.coef_
    max_theta_val.append(max(abs(theta)))
    deg_val.append(deg)

plt.scatter(deg_val, max_theta_val)
plt.xlabel('degrees')
plt.ylabel('max theta value')
plt.title('max theta value vs degrees')


    
    

