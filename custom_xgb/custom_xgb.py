import numpy as np 
from custom_xgb.node import TreeNode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

class CustomXGBoost(ClassifierMixin, BaseEstimator):
    
    def __init__(self, n_estimators, lr, reg_lambda, row_subsample_ratio, max_depth):
        self.lr = lr
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.row_subsample_ratio = row_subsample_ratio
        self.max_depth = max_depth
        
        self.params = {
            'lr': lr,
            'n_estimators': n_estimators,
            'reg_lambda': reg_lambda,
            'row_subsample_ratio': row_subsample_ratio,
            'max_depth': max_depth,
        }
        
        self.classes_ = np.array([0, 1])
        
    def set_params(self, **params):
        super().set_params(**params)
        
        self.params = params
        
        self.lr = params['lr']
        self.n_estimators = params['n_estimators']
        self.reg_lambda = params['reg_lambda']
        self.row_subsample_ratio = params['row_subsample_ratio']
        self.max_depth = params['max_depth']
        
        return self
                
    def fit(self, X, y):
        y_pred = np.zeros(shape=y.shape)
        self.estimators = []
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        for i in range(self.n_estimators):
            g = gradient(y, y_pred)
            h = hessian(y, y_pred)
            row_idxs = np.random.choice(
                X.shape[0], int(self.params['row_subsample_ratio']*X.shape[0]))
            
            tree = TreeNode()
            tree.fit(
                X, g, h, 
                row_idxs,
                self.max_depth,
                self.reg_lambda,)
            
            y_pred += self.lr * tree.predict(X)
            # print(f'Round {i}/{self.n_estimators}. Loss: {loss(y, y_pred)}')
            self.estimators.append(tree)
        self.estimator_ = self
        return self
            
    def predict(self, X):
        return sigmoid(self.lr * np.sum([tree.predict(X) for tree in self.estimators], axis=0)) >= 0.5
        
# def loss(y, y_pred): 
#     return np.mean((y - y_pred)**2)


# def gradient(y, y_pred): 
#     return y_pred - y


# def hessian(y, y_pred):
#     return np.ones(len(y))

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# negative log loss
def loss(y, y_pred): 
    y_pred = sigmoid(y_pred)
    return np.mean(- y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
    
def gradient(y, y_pred):
    return(sigmoid(y_pred) - y)

def hessian(y, y_pred):
    y_pred = sigmoid(y_pred)
    return(y_pred * (1 - y_pred))