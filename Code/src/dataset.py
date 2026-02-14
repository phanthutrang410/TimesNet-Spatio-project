
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from .utils import time_features

class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None,
                 features='M', target='OT', scale=True, freq='h', split_ratios=[0.7, 0.1, 0.2]):
        # Khởi tạo các tham số cơ bản
        self.seq_len = size[0]    # Độ dài chuỗi đầu vào
        self.label_len = size[1]  # Độ dài nhãn
        self.pred_len = size[2]   # Độ dài dự báo
        self.features = features  # Loại đặc trưng (M: Đa biến, S: Đơn biến)
        self.target = target      # Cột mục tiêu
        self.scale = scale        # Cờ chuẩn hóa dữ liệu
        self.root_path = root_path
        self.data_path = data_path
        self.freq = freq
        self.split_ratios = split_ratios
        
        # Mapping các tập dữ liệu: train=0, val=1, test=2
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Tính toán biên (borders) dựa trên tỷ lệ split
        if 'ETTh' in self.data_path:  # ETTh1 và ETTh2
            # Train: 12 months, Val: 4 months, Test: 4 months (đơn vị giờ)
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
            # border1s phải trừ seq_len để mẫu đầu Val/Test có đủ lịch sử
            border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
            border2s = [train_end, val_end, test_end]
            print(f"-> Standard Split (12/4/4 tháng): Train[0:{train_end}], Val[{border1s[1]}:{border2s[1]}], Test[{border1s[2]}:{border2s[2]}]")
        elif 'm' in self.data_path:
            # ETTm1, ETTm2: 15 minutes freq -> 1 hour = 4 points
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
            
            border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
            border2s = [train_end, val_end, test_end]
            print(f"-> ETTm Split (Fixed 15min): Train[0:{train_end}], Val[{border1s[1]}:{border2s[1]}], Test[{border1s[2]}:{border2s[2]}")
        else:
             num_train = int(len(df_raw) * self.split_ratios[0])
             num_test = int(len(df_raw) * self.split_ratios[2])
             num_vali = len(df_raw) - num_train - num_test
             border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
             border2s = [num_train, num_train + num_vali, len(df_raw)]
             print(f"-> Sử dụng Ratio Split: {self.split_ratios}")
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Lọc các cột dữ liệu cần thiết
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Chuẩn hóa dữ liệu (StandardScaler)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Xử lý đặc trưng thời gian (Time Features)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        # Trích xuất các thuộc tính thời gian (Giờ, Ngày, Thứ, Tháng...)
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        # data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Lấy mẫu dữ liệu theo index
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
