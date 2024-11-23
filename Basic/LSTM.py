import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class LSTM:
    def __init__(self, input_size, hidden_size):  # 修正为 __init__

        # 权重初始化（随机小值）
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_f = np.zeros((hidden_size, 1))

        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_i = np.zeros((hidden_size, 1))

        self.W_C = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_C = np.zeros((hidden_size, 1))

        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_o = np.zeros((hidden_size, 1))

        self.hidden_size = hidden_size

    def step(self, x_t, h_prev, C_prev):
        # 拼接输入和上一个隐藏状态
        combined = np.vstack((h_prev, x_t))
        # 计算遗忘门
        f_t = sigmoid(np.dot(self.W_f, combined) + self.b_f)
        # 计算输入门
        i_t = sigmoid(np.dot(self.W_i, combined) + self.b_i)
        # 生成候选记忆单元
        C_tilde = tanh(np.dot(self.W_C, combined) + self.b_C)

        # 更新记忆单元
        C_t = f_t * C_prev + i_t * C_tilde

        # 计算输出门
        o_t = sigmoid(np.dot(self.W_o, combined) + self.b_o)

        # 更新隐藏状态
        h_t = o_t * tanh(C_t)
        return h_t, C_t, f_t, i_t, o_t


# 设置时间步长和数据
time_steps = 100
x_data = np.sin(np.linspace(0, 3 * np.pi, time_steps))

# 初始化LSTM
input_size = 1
hidden_size = 10
lstm = LSTM(input_size=input_size, hidden_size=hidden_size)

# 初始化隐藏状态和记忆单元
h_t = np.zeros((hidden_size, 1))
C_t = np.zeros((hidden_size, 1))

# 保存门和状态的值以便后续绘图
h_states = []
C_states = []
f_gates = []
i_gates = []
o_gates = []

# 开始逐步处理时间序列
for t in range(time_steps):
    x_t = np.array([[x_data[t]]])  # 当前输入（维度为1）
    h_t, C_t, f_t, i_t, o_t = lstm.step(x_t, h_t, C_t)

    # 保存每个时刻的状态
    h_states.append(h_t)
    C_states.append(C_t)
    f_gates.append(f_t)
    i_gates.append(i_t)
    o_gates.append(o_t)

# 转换为NumPy数组以便绘图
h_states = np.squeeze(np.array(h_states))
C_states = np.squeeze(np.array(C_states))
f_gates = np.squeeze(np.array(f_gates))
i_gates = np.squeeze(np.array(i_gates))
o_gates = np.squeeze(np.array(o_gates))

# 开始绘图
plt.figure(figsize=(14, 12))

# 图1：输入数据的变化
plt.subplot(5, 1, 1)
plt.plot(x_data, label="Input (x_data)")
plt.title("Input Over Time")
plt.legend()

# 图2：隐藏状态的变化
plt.subplot(5, 1, 2)
for i in range(hidden_size):
    plt.plot(h_states[:, i], label=f'hidden state {i + 1}')
plt.title("Hidden State Over Time")
plt.legend()

# 图3：遗忘门的输出
plt.subplot(5, 1, 3)
for i in range(hidden_size):
    plt.plot(f_gates[:, i], label=f'forget gate {i + 1}')
plt.title("Forget Gate Over Time")
plt.legend()

# 图4：输入门的输出
plt.subplot(5, 1, 4)
for i in range(hidden_size):
    plt.plot(i_gates[:, i], label=f'input gate {i + 1}')
plt.title("Input Gate Over Time")
plt.legend()

# 图5：记忆单元状态变化
plt.subplot(5, 1, 5)
for i in range(hidden_size):
    plt.plot(C_states[:, i], label=f'cell state {i + 1}')
plt.title("Cell State Over Time")
plt.legend()

plt.tight_layout()
plt.show()
