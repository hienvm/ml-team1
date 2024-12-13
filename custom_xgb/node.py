import numpy as np 
from custom_xgb.split import find_split_exact_greedy, Split


class TreeNode():
 
    def __init__(
            self,
        ):
        self.split: Split | None = None
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None
        self.leaf_optimal_weight = 0
        

    def fit(
            self, 
            X: np.ndarray, 
            g: np.ndarray,
            h: np.ndarray,
            row_idxs: np.ndarray,
            remaining_depth,
            l2_lambda,
        ):
        if remaining_depth < 1:
            # tính giá trị tối ưu cho lá - Equation 5 https://arxiv.org/pdf/1603.02754
            self.leaf_optimal_weight = - g[row_idxs].sum() / (h[row_idxs].sum() + l2_lambda)
            return
        
        self.split = find_split_exact_greedy(
            X[row_idxs,:], g[row_idxs], h[row_idxs], l2_lambda)
        if self.split.col_idx is None:
            # tính giá trị tối ưu cho lá - Equation 5 https://arxiv.org/pdf/1603.02754
            self.leaf_optimal_weight = - g[row_idxs].sum() / (h[row_idxs].sum() + l2_lambda)
            return
        
        split_col = X[row_idxs, self.split.col_idx]
        
        left_idxs = np.nonzero(split_col < self.split.threshold)[0]
        right_idxs = np.nonzero(split_col >= self.split.threshold)[0]
        
        self.left = TreeNode()
        self.left.fit(
            X, g, h, 
            row_idxs[left_idxs], remaining_depth - 1, l2_lambda)
        
        self.right = TreeNode()
        self.right.fit(
            X, g, h, 
            row_idxs[right_idxs], remaining_depth - 1, l2_lambda)
        
                
                
    def predict(self, X: np.ndarray):
        if len(X.shape) == 1:
            # if single sample
            if self.left is None and self.right is None: 
                return self.leaf_optimal_weight
        
            if X[self.split.col_idx] < self.split.threshold:
                return self.left.predict(X)
            else:
                return self.right.predict(X)
            
        # if multiple samples
        return np.array([self.predict(sample) for sample in X])
    