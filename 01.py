# In[1]
import numpy as np
from scipy.sparse.linalg import svds

# 假设我们已经有一个用户-物品评分矩阵，用一个稀疏矩阵来表示
# 其中:行表示用户，列表示旅游线路，元素表示用户对该线路的评分
rating_matrix = np.array([[5, 3, 0, 1],
                          [4, 0, 4, 3],
                          [0, 1, 5, 4],
                          [2, 1, 3, 5]])

# 进行SVD分解，选择k个最大的奇异值
U, sigma, Vt = svds(rating_matrix, k=2)  # 选择2个潜在特征

# 将sigma对角化
sigma = np.diag(sigma)

# 计算预测评分
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

print(predicted_ratings)
