import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

np.random.seed(42)

N = 300
P = 10
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

fit_intercept = True

LR = LinearRegression(fit_intercept=fit_intercept)
start_1 = time.time()
LR.fit_autograd(X, y,15, lr_type = 'constant') ##here you can use fit_non_vectorised / fit_autograd methods
end_1 =time.time()
time_1 = end_1 - start_1

start_2 = time.time()
LR.fit_normal(X, y) ##for Normal Equation
end_2 = time.time()
time_2 = end_2 - start_2

print(time_1)
print(time_2)
