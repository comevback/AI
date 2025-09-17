import numpy as np

# 定义一个矩阵
A = np.array([[2, -1],
              [1, 1]])

# 求逆矩阵
A_inv = np.linalg.inv(A)

print("矩阵 A:")
print(A)

print("\nA 的逆矩阵 A_inv:")
print(A_inv)

# 验证 A * A_inv = I
I = np.dot(A, A_inv)
print("\nA * A_inv (应该接近单位矩阵):")
print(I)
