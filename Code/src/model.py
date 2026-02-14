import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CustomInceptionBlock, FFT_for_Period, ChannelAttention, CrossVariableAttention, GatedTemporalAttention
from .embed import DataEmbedding


class TimesBlockSpatio(nn.Module):
    """
    TimesBlock Spatio - Khối xử lý chính tích hợp Spatio-Temporal Attention.
    """
    def __init__(self, configs):
        super(TimesBlockSpatio, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.configs = configs

        # Mạng tích chập Inception
        self.conv = nn.Sequential(
            CustomInceptionBlock(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            CustomInceptionBlock(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

        # --- CÁC MODULE ATTENTION CẢI TIẾN (CÓ THỂ BẬT/TẮT) ---
        if getattr(configs, 'use_channel_attn', True):
            self.channel_attn = ChannelAttention(configs.d_model)
        
        if getattr(configs, 'use_cross_var_attn', False): # Mặc định TẮT để tránh overfitting
            self.cross_attn = CrossVariableAttention(configs.d_model, n_heads=4)
            
        if getattr(configs, 'use_gated_temporal', True):
            self.gated_attn = GatedTemporalAttention(configs.d_model, n_heads=4)

    def forward(self, x):
        B, T, N = x.size()
        
        # 1. Phân tích chu kỳ bằng FFT
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            
            # Padding nếu cần
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            # Reshape 1D -> 2D
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 2. Tích chập 2D
            out = self.conv(out)
            
            # [CẢI TIẾN 1] Áp dụng Channel Attention (Nếu bật)
            if hasattr(self, 'channel_attn'):
                out = self.channel_attn(out)
                
            # [CẢI TIẾN 2] Áp dụng Cross-Variable Attention (Nếu bật)
            if hasattr(self, 'cross_attn'):
                out = self.cross_attn(out)
                
            # Reshape lại 2D -> 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        res = torch.stack(res, dim=-1)
        
        # 3. Tổng hợp kết quả trọng số (Adaptive Aggregation)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # [CẢI TIẾN 3] Áp dụng Gated Temporal Attention (Nếu bật)
        if hasattr(self, 'gated_attn'):
            res = self.gated_attn(res)
        
        # Residual connection
        res = res + x
        return res


class TimesNetSpatio(nn.Module):
    """
    TimesNet Spatio - Mô hình chính tích hợp các cải tiến.
    """
    def __init__(self, configs):
        super(TimesNetSpatio, self).__init__()
        self.configs = configs
        self.model = nn.ModuleList([TimesBlockSpatio(configs) for _ in range(configs.e_layers)])
        
        # Embedding Layer
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        # Projection Layers
        self.predict_linear = nn.Linear(configs.seq_len, configs.pred_len + configs.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Normalization (RevIN)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesNet Blocks (với Spatio-Temporal Attention)
        for i in range(len(self.model)):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Projection
        dec_out = self.projection(enc_out)
        
        # De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len + self.configs.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len + self.configs.seq_len, 1)
        
        return dec_out[:, -self.configs.pred_len:, :]
