
import os
import argparse
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.model import TimesNetSpatio
from src.dataset import Dataset_Custom
from src.utils import EarlyStopping

def main():
    parser = argparse.ArgumentParser(description='TimesNet Spatio')

    # Cấu hình cơ bản
    parser.add_argument('--root_path', type=str, default='../Data/', help='Đường dẫn thư mục dữ liệu')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='Tên file dữ liệu')
    parser.add_argument('--is_training', type=int, default=1, help='Chế độ (1=Train, 0=Test)')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='TimesNet', help='model name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Tham số dữ liệu
    parser.add_argument('--seq_len', type=int, default=96, help='Độ dài chuỗi đầu vào')
    parser.add_argument('--label_len', type=int, default=48, help='Độ dài nhãn')
    parser.add_argument('--pred_len', type=int, default=96, help='Độ dài dự báo')
    parser.add_argument('--features', type=str, default='M', help='Loại dự báo: M=đa biến, S=đơn biến, MS=đa-đơn')
    parser.add_argument('--target', type=str, default='OT', help='Cột mục tiêu (cho chế độ S hoặc MS)')
    parser.add_argument('--freq', type=str, default='h', help='Tần suất dữ liệu: h=giờ, d=ngày, m=tháng')

    # Tham số mô hình
    parser.add_argument('--top_k', type=int, default=5, help='Top-k tần số cho FFT')
    parser.add_argument('--num_kernels', type=int, default=6, help='Số kernel Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='Số đặc trưng đầu vào')
    parser.add_argument('--dec_in', type=int, default=7, help='Số đặc trưng decoder')
    parser.add_argument('--c_out', type=int, default=7, help='Số đặc trưng đầu ra')
    parser.add_argument('--d_model', type=int, default=32, help='Kích thước vector ẩn')
    parser.add_argument('--d_ff', type=int, default=32, help='Kích thước FeedForward')
    parser.add_argument('--e_layers', type=int, default=2, help='Số lớp Encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='Tỷ lệ Dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='Loại embedding thời gian')

    # Tham số huấn luyện
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    # Cờ kích hoạt các module cải tiến (Spatio)
    parser.add_argument('--use_channel_attn', action='store_true', default=True, help='Bật Channel Attention')
    parser.add_argument('--use_cross_var_attn', action='store_true', default=True, help='Bật Cross-Variable Attention')
    parser.add_argument('--use_gated_temporal', action='store_true', default=True, help='Bật Gated Temporal Attention')

    args = parser.parse_args()

    # --- AUTO-DETECT FEATURES ---
    # Đọc tiêu đề CSV để tự động xác định số lượng feature
    file_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(file_path):
         # Thử tải dữ liệu mẫu nếu chưa có
        if not os.path.exists(args.root_path):
            os.makedirs(args.root_path)
            
        print(f"File không tồn tại: {file_path}")
        print(f"Đang tải dữ liệu mẫu ETTh1 về {file_path}...")
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        import urllib.request
        try:
            urllib.request.urlretrieve(url, file_path)
            print("Tải xuống hoàn tất.")
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return
            
    try:
        df = pd.read_csv(file_path, nrows=1)
        num_cols = len(df.columns)
        # Giả sử cột đầu tiên là date, các cột còn lại là features
        detected_features = num_cols - 1 
        
        print(f"--> Tự động phát hiện {detected_features} đặc trưng từ file {args.data_path}")
        
        if args.features == 'M' or args.features == 'MS':
            args.enc_in = detected_features
            args.c_out = detected_features
        else: # S (Univariate)
            args.enc_in = 1
            args.c_out = 1
            
        print(f"--> Đã cập nhật: enc_in={args.enc_in}, c_out={args.c_out}")
            
    except Exception as e:
        print(f"Không thể tự động đọc số lượng feature: {e}")
        print("Sử dụng giá trị mặc định.")

    # Thiết lập thiết bị
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Hệ thống đang sử dụng: {device}")

    # 1. Dataset & Dataloader
    print("Đang khởi tạo Dataset...")
    train_dataset = Dataset_Custom(
        root_path=args.root_path, data_path=args.data_path, flag='train',
        size=[args.seq_len, args.label_len, args.pred_len], 
        features=args.features, target=args.target, freq=args.freq
    )
    val_dataset = Dataset_Custom(
        root_path=args.root_path, data_path=args.data_path, flag='val',
        size=[args.seq_len, args.label_len, args.pred_len], 
        features=args.features, target=args.target, freq=args.freq
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    test_dataset = Dataset_Custom(
        root_path=args.root_path, data_path=args.data_path, flag='test',
        size=[args.seq_len, args.label_len, args.pred_len], 
        features=args.features, target=args.target, freq=args.freq
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. Khởi tạo Mô hình
    print("Đang xây dựng Mô hình...")
    model = TimesNetSpatio(args).to(device)
    print(f"Cấu hình cải tiến: Channel={args.use_channel_attn}, Cross={args.use_cross_var_attn}, Gated={args.use_gated_temporal}")
    
    # 3. Huấn luyện (Training)
    if args.is_training:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Checkpoint path
        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints)
        
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        print("Bắt đầu quá trình Huấn luyện...")
        for epoch in range(args.train_epochs):
            model.train()
            train_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                
                outputs = model(batch_x, batch_x_mark)
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                
            train_loss_avg = np.average(train_loss)
            
            # Validation
            model.eval()
            val_loss = []
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    
                    outputs = model(batch_x, batch_x_mark)
                    
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    
                    loss = criterion(outputs, batch_y)
                    val_loss.append(loss.item())
                    
            val_loss_avg = np.average(val_loss)
            print(f"Epoch: {epoch+1} | Train Loss: {train_loss_avg:.5f} | Val Loss: {val_loss_avg:.5f}")
            
            # Early Stopping Check
            early_stopping(val_loss_avg, model, args.checkpoints)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
                
        # Load best model for testing
        print("Loading best model for testing...")
        best_model_path = os.path.join(args.checkpoints, 'checkpoint.pth')
        model.load_state_dict(torch.load(best_model_path))

    # 4. Đánh giá (Testing)
    print(">>>>>>> Testing <<<<<<<")
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            
            outputs = model(batch_x, batch_x_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    print(f"Test Shape: {preds.shape}")
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues)**2)
    
    print(f"MSE: {mse:.5f}, MAE: {mae:.5f}")

if __name__ == '__main__':
    main()
