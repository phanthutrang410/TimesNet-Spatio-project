import torch
import torch.nn as nn
import torch.fft

# --- 1. Custom Inception Block (Khối Inception Tùy chỉnh) ---
class CustomInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super(CustomInceptionBlock, self).__init__()
        kernels = []
        # Tạo nhiều kernel với kích thước khác nhau để bắt đa dạng mẫu hình
        for i in range(num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x):
        res_list = []
        for i in range(len(self.kernels)):
            res_list.append(self.kernels[i](x))
        # Tổng hợp kết quả từ các kernel bằng phép lấy trung bình
        return torch.stack(res_list, dim=-1).mean(-1)

# --- 2. FFT Logic (Biến đổi Fourier nhanh) ---
def FFT_for_Period(x, k=2):
    # Biến đổi Fourier rFFT
    xf = torch.fft.rfft(x, dim=1)
    # Tính tần số trung bình
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    # Chọn Top-k tần số quan trọng nhất
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    # Tính chu kỳ tương ứng
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

# --- 3. Attention Modules (Các Module Cải tiến) ---

# [Module 1] Channel Attention (Chú ý Kênh)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        self.attn_weights = None  # Placeholder output visualization

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        self.attn_weights = y.view(b, c).detach()  # Save for visualization
        return x * y.expand_as(x)

# [Module 2] Cross-Variable Attention (Chú ý Đa biến)
class CrossVariableAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(CrossVariableAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten để đưa vào cơ chế Attention
        x_flatten = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        attn_out, weights = self.attn(x_flatten, x_flatten, x_flatten) # Capture weights
        self.attn_weights = weights.detach() # Save for visualization

        out = x_flatten + self.dropout(attn_out)
        out = self.norm(out)
        return out.view(B, H, W, C).permute(0, 3, 1, 2)

# [Module 3] Gated Temporal Attention (Cổng Chú ý Thời gian)
class GatedTemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(GatedTemporalAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate_weights = None

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout(attn_out)
        # Cơ chế cổng (Gating mechanism)
        gate_input = torch.cat([x, attn_out], dim=-1)
        gate = self.gate_fc(gate_input)
        self.gate_weights = gate.detach().squeeze(-1) # Save for visualization [Seq_len]

        out = gate * attn_out + (1 - gate) * x
        return self.norm(out)
