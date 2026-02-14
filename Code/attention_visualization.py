# ============================================================
# ATTENTION VISUALIZATION CODE
# Copy toàn bộ code này vào một cell MỚI trong notebook
# Chạy SAU KHI train xong
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention_weights(model, test_loader, device, data_name='ETTh1'):
    """
    Visualize Channel Attention và Gated Temporal Attention weights
    
    Sử dụng: visualize_attention_weights(model, test_loader, device, 'ETTh1.csv')
    """
    model.eval()
    
    # Lấy 1 batch từ test
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    
    # Forward pass để tính attention weights
    with torch.no_grad():
        _ = model(batch_x, batch_x_mark)
    
    # Lấy weights từ layer đầu tiên
    try:
        channel_weights = model.model[0].channel_attn.attn_weights[0].cpu().numpy()
    except:
        channel_weights = None
        print("Không có Channel Attention weights")
    
    try:
        gated_weights = model.model[0].gated_attn.gate_weights[0].cpu().numpy()
    except:
        gated_weights = None
        print("Không có Gated Temporal weights")
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Attention Weights Visualization - {data_name}', fontsize=14)
    
    # ===== 1. Channel Attention =====
    if channel_weights is not None:
        channel_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        colors = ['steelblue'] * 6 + ['coral']  # OT màu khác
        
        bars = axes[0].bar(range(len(channel_weights)), channel_weights, color=colors)
        axes[0].set_xlabel('Biến đầu vào', fontsize=12)
        axes[0].set_ylabel('Trọng số Attention', fontsize=12)
        axes[0].set_title('Channel Attention\n(Biến nào quan trọng nhất?)', fontsize=12)
        axes[0].set_xticks(range(len(channel_weights)))
        axes[0].set_xticklabels(channel_names[:len(channel_weights)], rotation=45)
        axes[0].axhline(y=np.mean(channel_weights), color='red', linestyle='--', alpha=0.7, label='Trung bình')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Không có Channel Attention', ha='center', va='center')
    
    # ===== 2. Gated Temporal Attention =====
    if gated_weights is not None:
        axes[1].plot(gated_weights, color='coral', linewidth=2)
        axes[1].fill_between(range(len(gated_weights)), gated_weights, alpha=0.3, color='coral')
        axes[1].set_xlabel('Bước thời gian (Time Step)', fontsize=12)
        axes[1].set_ylabel('Giá trị Cổng (Gate)', fontsize=12)
        axes[1].set_title('Gated Temporal Attention\n(0=Giữ nguyên, 1=Dùng Attention)', fontsize=12)
        axes[1].set_ylim(0, 1)
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Ngưỡng 0.5')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'Không có Gated Temporal', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f'attention_{data_name.replace(".csv", "")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Đã lưu visualization tại: attention_{data_name.replace('.csv', '')}.png")
    
    # In thông tin chi tiết
    if channel_weights is not None:
        print("\n📊 Channel Attention Weights:")
        channel_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        for i, w in enumerate(channel_weights):
            name = channel_names[i] if i < len(channel_names) else f"Channel {i}"
            print(f"   {name}: {w:.4f}")
        print(f"   → Biến quan trọng nhất: {channel_names[np.argmax(channel_weights)]}")
    
    if gated_weights is not None:
        print(f"\n📊 Gated Temporal Stats:")
        print(f"   Mean: {np.mean(gated_weights):.4f}")
        print(f"   Min: {np.min(gated_weights):.4f}")
        print(f"   Max: {np.max(gated_weights):.4f}")


# ============================================================
# CÁCH SỬ DỤNG:
# ============================================================
# Thêm dòng này vào cuối vòng lặp benchmark, sau khi train xong:
#
# for data_name in datasets_to_run:
#     ...
#     model = train_model(...)
#     ...
#     # ===== THÊM DÒNG NÀY =====
#     visualize_attention_weights(model, test_loader, device, data_name)
#     # =========================
#
# ============================================================
