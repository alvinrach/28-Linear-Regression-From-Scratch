#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class LinReg:
    """
    Linear regression class to predict coefficient and constant of single or multi-variable data.
    Parameters
    ----------
    lr : float, default=0.01
        Set the learning rate.
    percent_diff : float, default=0.1
        Set the difference percent before the epoch stops.
    Attributes
    ----------
    coefs : array of shape (n_features, ) or (n_targets, n_features)
        Return slope of the Linear Regression model object.
    const : float
        Retrun intercept of the Linear Regression model object.
    losses : array of shape (min(X, y),)
        Return loss progress each epoch of the model object.
    epochs : int
        Return how many epoch has applied when fit of the model object.
    Examples
    --------
    Input:
        import numpy as np
        from LinearReg import LinReg
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        # y = 1 * x_0 + 2 * x_1 + 3
        y = np.dot(X, np.array([1, 2])) + 3
        reg = LinReg().fit(X, y)
        reg.coefs
    Output:
    array([[1.],
           [2.]])
           
    Input:
        reg.const
    Output:
    2.999999999989651
    
    Input:
        reg.epochs
    Output:
    35560
    
    Input:
        reg.predict(np.array([[3, 5], [4, 6]]))
    Output:
    array([[16.],
           [19.]])
           
    Input:
        reg.mae(test=np.array([[20], [25]]), pred=reg.predict(np.array([[3, 5], [4, 6]])).flatten())
    Output:
    array([6.5, 3.5])
    
    Input:
        reg.rmse(test=np.array([[20], [25]]), pred=reg.predict(np.array([[3, 5], [4, 6]])).flatten(), unroot=True)
    Output:
    array([48.5, 18.5])
    """
    def __init__(self, lr=0.01, percent_diff = 0.1):
        self.lr = lr
        self.diff = percent_diff
        
    def fit(self, X, y):
        """
        Fit linear model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input variables values.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target variable values.
        Returns
        -------
        self : returns an instance of self.
        """
        m, n = X.shape    
        
        self.coefs = np.zeros((n,1))
        self.const = 0
        
        y = y.reshape(m,1)
        
        self.losses = []
        
        diff = 100
        
        while diff>=self.diff:
            
            y_temp = np.dot(X, self.coefs) + self.const
            
            loss = np.mean((y_temp - y)**2)
            self.losses.append(loss)

            d_coef = (1/m)*np.dot(X.T, (y_temp - y))
            d_const = (1/m)*np.sum((y_temp - y))
            
            self.coefs -= self.lr*d_coef
            self.const -= self.lr*d_const
            
            if len(self.losses)>=2:
                diff=(self.losses[-2]-self.losses[-1])*100/self.losses[-2]
        
        self.epochs=len(self.losses)
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Data to predict.
        Returns
        -------
        array, shape (n_samples,)
            Returns predicted values.
        """
        return np.dot(X, self.coefs) + self.const
    
    def mae(self, test, pred):
        """Mean absolute error regression loss of the linear model.
        Parameters
        ----------
        test : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Correct target values.
        pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.
        Returns
        -------
        float or ndarray of floats
            A non-negative floating point. The best value is 0.0.
        """
        b=len(test)
        return sum(abs(np.subtract(test, pred)))/b
    
    def rmse(self, test, pred, unroot=False):
        """Root Mean squared error regression loss of the linear model.
        Parameters
        ----------
        test : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Correct target values.
        pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.
        unroot : bool, default=False
            If True, prediction will return Mean Squared Error.
        Returns
        -------
        float or ndarray of floats
            A non-negative floating point. The best value is 0.0.
        """
        b=len(test)
        
        if not unroot:
            return np.sqrt(sum(np.subtract(test, pred)**2)/b)
        else:
            return sum(np.subtract(test, pred)**2)/b

