
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_autograd(X, y,15, lr_type = 'constant') # here you can use fit_non_vectorised / fit_autograd methods
    X = pd.DataFrame(np.random.randn(N, P))
    y_hat = LR.predict(X)
    print(y_hat)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
