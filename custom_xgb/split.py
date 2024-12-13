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
        # Chỉ chọn từ sqrt(n) feature ngẫu nhiên
        col_idxs = np.random.choice(
            X.shape[1], int(np.sqrt(X.shape[1])), replace=False)
        for k in col_idxs: 
            # Sắp xếp các sample theo thứ tự tăng dần của feature thứ k
            sorted_row_idxs = X[:,k].argsort()
            GL, HL, GR, HR = 0, 0, G, H
            for i in range(len(sorted_row_idxs) - 1):
                # Đẩy threshold từ trái sang
                j = sorted_row_idxs[i]
                j_next = sorted_row_idxs[i + 1]
                GL += g[j]
                HL += h[j]
                GR -= g[j]
                HR -= h[j]
                # Cập nhật score (tỉ lệ với split gain)
                score = GL**2 / (HL + l2_lambda) \
                        + GR**2 / (HR + l2_lambda) \
                        - G**2 / (H + l2_lambda)
                if score > best_split.score: 
                    best_split.score = score
                    best_split.col_idx = k
                    # Đặt threshold ở giữa 2 phần tử liền kề
                    best_split.threshold = (X[j, k] + X[j_next, k]) / 2.0
        return best_split