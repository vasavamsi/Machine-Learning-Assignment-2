import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

x_1 = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

x_2 = x_1*2 ##Taking second feature as multiple of first feature for introducing multicolinearity
data = pd.DataFrame({'x_1':x_1, 'x_2':x_2})
y = pd.Series(y)

for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_autograd(data, y,1,n_iter=300, lr_type = 'constant') # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(data)

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
print('Yes it works for Multicolinear dataset')
    


