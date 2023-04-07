import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

X = pd.DataFrame(x)
y = pd.Series(y)

LR = LinearRegression(fit_intercept=True)
t_0,t_1 = LR.fit_autograd(X, y,1, lr_type = 'constant') ##We are plotting on autograd fitted model

#LR.plot_line_fit(X,y,t_0,t_1)
#LR.plot_contour(X, y, t_0, t_1)
LR.plot_surface(X, y, t_0, t_1)
