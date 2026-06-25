import torch
import torch.nn as nn
import torch.optim as optim
import copy


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FLClient:
    def __init__(self, client_id, dataloader, device='cpu'):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.model = SimpleMLP().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # 模型体积（假设 float32 -> 4 bytes）
        self.base_model_size_bytes = sum(p.numel() for p in self.model.parameters()) * 4

    def update_model(self, global_state_dict):
        """用服务器下发的全局模型更新本地模型"""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def compute_communication_bytes(self, quantization):
        """
        计算本次通信的（逻辑）字节开销
        :param quantization: 'none', '8-bit', '4-bit'
        """
        sparsity_ratio = 0.9 if quantization != 'none' else 1.0
        if quantization == '4-bit':
            bytes_sent = (self.base_model_size_bytes / 8.0) * sparsity_ratio
        elif quantization == '8-bit':
            bytes_sent = (self.base_model_size_bytes / 4.0) * sparsity_ratio
        else:
            bytes_sent = self.base_model_size_bytes
        return float(bytes_sent)

    # ===================== 真实的压缩实现（修复“压缩不生效”）=====================
    @staticmethod
    def _topk_sparsify(t, keep_ratio):
        """Top-k 稀疏化：仅保留幅值最大的 keep_ratio 比例，其余置零。"""
        n = t.numel()
        if n == 0 or keep_ratio >= 1.0:
            return t
        k = max(1, int(n * keep_ratio))
        if k >= n:
            return t
        flat = t.flatten()
        thresh = torch.kthvalue(flat.abs(), n - k).values
        mask = (flat.abs() >= thresh).to(t.dtype)
        return (flat * mask).view_as(t)

    @staticmethod
    def _quantize_tensor(t, bits):
        """逐张量均匀量化到 bits 比特，再反量化（引入真实的量化误差）。"""
        if t.numel() == 0:
            return t
        qmin, qmax = t.min(), t.max()
        if (qmax - qmin) < 1e-12:
            return t.clone()
        levels = (2 ** bits) - 1
        scale = (qmax - qmin) / levels
        q = torch.round((t - qmin) / scale)
        return q * scale + qmin

    def compress_delta(self, global_state, local_state, quantization, sparsity_keep=0.9):
        """
        对“本地权重 - 全局权重”的增量做 Top-k 稀疏化 + k-bit 量化，再加回全局权重重建。
        这才是真正作用到传输内容上的压缩：会真实降低通信量，并带来可见的小幅精度代价。
        """
        if quantization == 'none':
            return local_state
        bits = 4 if quantization == '4-bit' else 8
        out = {}
        for k in local_state:
            g, l = global_state[k], local_state[k]
            if l.is_floating_point():
                delta = self._topk_sparsify(l - g, sparsity_keep)
                delta = self._quantize_tensor(delta, bits)
                out[k] = g + delta
            else:
                out[k] = l
        return out

    def train(self, epochs, learning_rate=0.01, dp_sigma=0.0, dp_clip=1.0):
        """
        本地训练
        :param dp_sigma: 差分隐私高斯噪声标准差（=0 表示不加 DP）
        :param dp_clip:  梯度裁剪阈值
        :return: (更新后的状态字典, 本地样本数, 训练损失)
        """
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        epoch_loss = []
        if epochs == 0:
            return copy.deepcopy(self.model.state_dict()), len(self.dataloader.dataset), 0.0

        for epoch in range(epochs):
            batch_loss = []
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # 差分隐私：梯度裁剪 + 加高斯噪声
                if dp_sigma > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=dp_clip)
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad += torch.randn_like(param.grad) * dp_sigma

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        num_samples = len(self.dataloader.dataset)
        return copy.deepcopy(self.model.state_dict()), num_samples, avg_loss