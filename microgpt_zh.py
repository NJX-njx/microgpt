"""
以最原子化的方式，在纯粹、无依赖的 Python 中训练和推理 GPT。
此文件包含完整的算法。
其他一切都只是为了效率。

@karpathy (优化版)
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
import time     # time.time

# 在混沌中建立秩序
random.seed(42)

# 准备输入数据集 `docs`：字符串列表（例如：名字数据集）
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
with open('input.txt') as f: # 正确关闭文件句柄
    docs = [l.strip() for l in f.read().strip().split('\n') if l.strip()]
random.shuffle(docs)
split = int(0.9 * len(docs))
train_docs, val_docs = docs[:split], docs[split:]
print(f"num docs: {len(docs)} (train: {len(train_docs)}, val: {len(val_docs)})")

# 定义分词器（Tokenizer），用于将字符串翻译为离散符号，反之亦然
chars = ['<BOS>'] + sorted(set(''.join(docs))) # 带有 BOS 分隔符的字符级分词器
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) } # 编码：映射字符串到整数
itos = { i:ch for i, ch in enumerate(chars) } # 解码：映射整数到字符串
BOS = stoi['<BOS>']
print(f"vocab size: {vocab_size}")

# 定义自动求导（Autograd），通过计算图递归应用链式法则
# 从而计算损失函数相对于模型参数的梯度。
class Value:
    """存储单个标量值及其梯度。"""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # 产生此节点的算子，用于绘图/调试等

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "目前仅支持整数/浮点数指数"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other): # 直接除法：比 self * other**-1 产生的图节点更少
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self): # 迭代拓扑排序（无递归深度限制）
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): # 现在使用直接的 __truediv__
        other = other if isinstance(other, Value) else Value(other)
        return other / self
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

# 初始化参数，用于存储模型的知识。
n_embd = 16     # 嵌入维度
n_head = 4      # 注意力头数
n_layer = 1     # 层数
block_size = 8  # 最大序列长度
head_dim = n_embd // n_head # 每个注意力头的维度
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd)} # 权重绑定：wte 同时作为 lm_head
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)
params = [p for mat in state_dict.values() for row in mat for p in row] # 将参数打平为单个 list[Value]
print(f"num params: {len(params)}")

# 定义模型架构：一个无状态函数，将 token 序列和参数映射为下一个可能 token 的 logits。
# 遵循经典的 GPT-2 架构，仅有细微差别：layernorm -> rmsnorm, 无 bias, GeLU -> ReLU^2
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def cross_entropy(logits, target_id): # 融合了 log-softmax + nll：节点更少，数值更稳定
    max_val = max(val.data for val in logits)
    shifted = [val - max_val for val in logits]
    log_sum_exp = sum(s.exp() for s in shifted).log()
    return -(shifted[target_id] - log_sum_exp)

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token 嵌入
    pos_emb = state_dict['wpe'][pos_id] # 位置嵌入
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # 合并 token 和位置嵌入
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) 多头注意力模块
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP 模块
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() ** 2 for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['wte']) # 权重绑定：复用 token 嵌入作为输出投影
    return logits

# 定义 Adam 优化器及其缓存
learning_rate, beta1, beta2, eps_adam, weight_decay = 1e-2, 0.9, 0.95, 1e-8, 1e-4
m = [0.0] * len(params) # 一阶矩缓存
v = [0.0] * len(params) # 二阶矩缓存
b1_prod, b2_prod = 1.0, 1.0 # 用于偏置校正的累乘值
max_grad_norm = 1.0 # 梯度裁剪阈值

# 循环训练
num_steps = 500 # 训练步骤数
for step in range(num_steps):
    t0 = time.time()

    # 取单个文档，进行分词，并在两端加上 BOS 特殊 token
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # 前向传播：将 token 序列通过模型，构建通向损失函数的计算图
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        loss_t = cross_entropy(logits, target_id) # 融合的交叉熵替代了 softmax+log
        losses.append(loss_t)
    loss = (1 / n) * sum(losses[1:], losses[0]) # 避免产生幻影 Value(0) 节点

    # 反向传播：计算损失函数相对于所有模型参数的梯度
    loss.backward()

    # 按全局范数进行梯度裁剪
    grad_norm = sum(p.grad ** 2 for p in params) ** 0.5
    if grad_norm > max_grad_norm:
        for p in params: p.grad *= max_grad_norm / grad_norm

    # 使用余弦学习率调度的 AdamW 优化器更新
    lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    b1_prod *= beta1; b2_prod *= beta2 # 使用累乘代替 beta**step
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - b1_prod) # 稳定的偏置校正
        v_hat = v[i] / (1 - b2_prod)
        p.data -= lr_t * (m_hat / (v_hat ** 0.5 + eps_adam) + weight_decay * p.data)
        p.grad = 0

    print(f"step {step+1:4d}/{num_steps:4d} | loss {loss.data:.4f} | {(time.time()-t0)*1000:.0f}ms")

    # 定期在验证集文档上进行验证
    if (step + 1) % 100 == 0:
        val_loss, val_n = 0.0, 0
        for vdoc in val_docs[:20]:
            vt = [BOS] + [stoi[ch] for ch in vdoc] + [BOS]
            vn = min(block_size, len(vt) - 1)
            vkeys, vvals = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
            for vp in range(vn):
                vlogits = gpt(vt[vp], vp, vkeys, vvals)
                mx = max(l.data for l in vlogits) # 纯浮点 NLL，无自动求导开销
                val_loss -= vlogits[vt[vp+1]].data - mx - math.log(sum(math.exp(l.data - mx) for l in vlogits))
                val_n += 1
        print(f"  val loss: {val_loss / val_n:.4f}")

# 推理：让模型开始“胡言乱语”，使用 top-k 采样
temperature, top_k = 0.6, 5
print("\n--- 推理 ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    print(f"sample {sample_idx+1:2d}: ", end="")
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        top_ids = sorted(range(vocab_size), key=lambda i: logits[i].data, reverse=True)[:top_k]
        top_logits = [logits[i].data / temperature for i in top_ids]
        max_tl = max(top_logits)
        top_probs = [math.exp(tl - max_tl) for tl in top_logits]
        token_id = top_ids[random.choices(range(top_k), weights=top_probs)[0]]
        if token_id == BOS:
            break
        print(itos[token_id], end="")
    print()
