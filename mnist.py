# COMMAND ----------
import struct
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------


def load_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number mismatch, got {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # flat the data to N * 784 matrix and change to float32
        images = (data.reshape(num, rows*cols)).astype(np.float32) / 255.0
        return images

# COMMAND ----------


def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number mismatch, got {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

# COMMAND ----------


def one_hot(labels: np.ndarray):
    targets = np.zeros((labels.size, 10), dtype=np.float32)
    # 生成一个长度为labels长度的索引数列
    index_matrix = np.arange(labels.size)
    # 用labels向量的个数来做行索引，用它的值来做列索引
    targets[index_matrix, labels] = 1
    return targets

# COMMAND ----------
# 前向传播函数，得到Z


def forward(X: np.ndarray, W: np.ndarray, b: np.ndarray):
    Z = X.dot(W) + b
    return Z

# COMMAND ----------
# softmax处理，把Z变成概率矩阵


def softMax(Z: np.ndarray):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    return probs

# COMMAND ----------
# 计算损失函数的Y_onehot版本


def cross_entropy_from_onehot(Y_hat: np.ndarray, Y_onehot: np.ndarray, eps=1e-12):
    logp = np.log(Y_hat + eps)  # 给矩阵每个元素加一个微小的eps，避免无穷，然后把每个数变成log
    # 逐元素相乘：把 logp 和 Y_onehot 对位相乘。
    # 把矩阵所有值相加，但是不为零的只有对应位置，所以是所有正确概率log相加除以个数
    loss = -np.sum(Y_onehot * logp) / Y_hat.shape[0]
    # 得到的就是损失函数
    return loss

# COMMAND ----------
# 计算损失函数的y_int版本


def cross_entropy_from_int(Y_hat: np.ndarray, y_int: np.ndarray, eps: int = 1e-12):
    # 生成一个y_hat长度的index索引
    rows = np.arange(Y_hat.shape[0])
    # 使用高级索引，得到y_hat中所有需要得到的值，也就是对应正确答案的概率
    p_true = Y_hat[rows, y_int]
    # 把所有正确答案加eps求log，然后求负平均值
    loss = -np.mean(np.log(p_true + eps))
    return loss

# COMMAND ----------
# 计算损失函数对Zt的导数，作为求得梯度的前提


def d_loss_d_Z(Y_hat: np.ndarray, Y_onehot: np.ndarray, B: int):
    G = Y_hat.copy()
    G -= Y_onehot
    G /= B
    return G


# COMMAND ----------
# 读取文件
images = load_idx_images("./train-images.idx3-ubyte")
labels = load_idx_labels("./train-labels.idx1-ubyte")

# 设定训练样本大小
batch_size = 128
# 设定训练轮次
epochs = 20
# 设定学习率
N = images.shape[0]
lr = 0.1

# COMMAND ----------
# 创建参数矩阵
rows, cols = 784, 10
W = (np.random.randn(rows, cols) * 0.01).astype(np.float32)
b = np.zeros((1, 10), dtype=np.float32)

losses = []
accuracies = []

# COMMAND ----------
for epoch in range(epochs):
    indices = np.random.permutation(N)
    X_shuffled = images[indices]
    Y_shuffled = labels[indices]

    for start in range(0, N, batch_size):
        end = start + batch_size
        Xb = X_shuffled[start:end]
        Yb = Y_shuffled[start:end]
        B = Xb.shape[0]

        # forward
        Z = Xb @ W + b
        probs = softMax(Z)

        # reverse
        Y_onehot = one_hot(Yb)
        G = (probs - Y_onehot) / B
        grad_W = Xb.T @ G
        grad_b = G.sum(axis=0, keepdims=True)

        # update W and b
        W -= lr * grad_W
        b -= lr * grad_b

    Z_all = images @ W + b
    probs_all = softMax(Z_all)
    loss = cross_entropy_from_int(probs_all, labels)
    acc = (probs_all.argmax(axis=1) == labels).mean()
    losses.append(loss)
    accuracies.append(acc)
    print(f"round {epoch + 1}: loss={loss:.4f}, acc={acc:.4f}")

# COMMAND ----------
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), losses, label='Loss', marker='o')
plt.plot(range(1, epochs + 1), accuracies, label='Accuracy', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()

for i, (l, a) in enumerate(zip(losses, accuracies)):
    plt.text(i + 1, l, f"{l:.3f}", ha='left', va='bottom', fontsize=8)
    plt.text(i + 1, a, f"{a:.3f}", ha='center', va='bottom', fontsize=8)
plt.show()

# COMMAND ----------
X_test_path = "./t10k-images.idx3-ubyte"
Y_test_path = "./t10k-labels.idx1-ubyte"
X_test = load_idx_images(X_test_path)
Y_test = load_idx_labels(Y_test_path)

Z_test = X_test @ W + b
probs_test = softMax(Z_test)
loss_test = cross_entropy_from_int(probs_test, Y_test)
acc_test = (probs_test.argmax(axis=1) == Y_test).mean()
print(f"\ntest result: loss: {loss_test:4f}, acc: {acc_test:4f}.")
np.save("W.npy", W)
np.save("b.npy", b)
