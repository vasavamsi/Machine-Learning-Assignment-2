import numpy as np
from decimal import *

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    coun = 0
    y = y.values
    y_hat = y_hat.values
    for i in range(np.shape(y)[0]):
        if y_hat[i] == y[i]:
            coun += 1
    acc = Decimal(coun)/Decimal(np.shape(y)[0])
    return float(acc)

    #pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """

    coun = 0
    count_y_hat = 0

    for i in range(y.size):
        if y_hat[i] == cls:
            count_y_hat += 1
        if y_hat[i] == cls and y[i] == cls:
            coun += 1

    precise = Decimal(coun)/Decimal(count_y_hat)
    return float(precise)

    #pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    coun = 0
    count_y_cls = 0
    for i in range(y.size):
        if y[i] == cls:
            count_y_cls += 1
        if y_hat[i] == cls and y[i] == cls:
            coun += 1
    output = Decimal(coun)/Decimal(count_y_cls)
    return float(output)
    #pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    output = (float(sum((y-y_hat)**2))/float(y.size))**0.5
    return float(output)
    #pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    output = (sum(abs(y-y_hat)))/y.size
    return float(output)

    #pass
