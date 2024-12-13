import numpy as np

class Split:
    def __init__(self):
        self.score: float = 0.0
        self.col_idx: int | None = None
        self.threshold: float | None = None

def find_split_exact_greedy(X, g, h, l2_lambda) -> 'Split':
        best_split = Split()
        G = g.sum()
        H = h.sum()
        
        col_idxs = np.random.choice(
            X.shape[1], int(np.sqrt(X.shape[1])), replace=False)
        
        for k in col_idxs: 
            sorted_row_idxs = X[:,k].argsort()
            
            GL = 0
            HL = 0
            GR = G
            HR = H

            for i in range(len(sorted_row_idxs) - 1):
                j = sorted_row_idxs[i]
                j_next = sorted_row_idxs[i + 1]
                
                GL += g[j]
                HL += h[j]
                
                GR -= g[j]
                HR -= h[j]

                score = GL**2 / (HL + l2_lambda) \
                        + GR**2 / (HR + l2_lambda) \
                        - G**2 / (H + l2_lambda)
                
                if score > best_split.score: 
                    best_split.score = score
                    best_split.col_idx = k
                    best_split.threshold = (X[j, k] + X[j_next, k]) / 2.0
                    
        return best_split