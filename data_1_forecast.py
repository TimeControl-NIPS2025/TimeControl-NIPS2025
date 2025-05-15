# -*- coding: utf-8 -*

import os
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from distutils.util import strtobool

import warnings
warnings.filterwarnings('ignore')


def convert_tsf_to_dataframe(
        full_file_path_and_name,
        replace_missing_vals_with="NaN",
        value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                    len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                    len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                                numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


"""
long_term_forecast / imputation
【self.data_x】
1、基于self.root_path和self.data_path从指定的csv文件中读取pandas对象df_raw
2、随后从df_raw中获取指定列作为df_data，并对df_data进行标准化处理得到data
3、最后从data中读取lookback和forecast

# #                           ETTh1-pretrain     ETTh1-test      ETTh2-pretrain     ETTh2-test
# print(self.data.shape)      # (17420, 7)    (17420, 7)      (17420, 7)      (17420, 7)
# print(self.data_x.shape)    # (8640, 7)     (2976, 7)       (8640, 7)       (2976, 7)
# print(self.data_y.shape)    # (8640, 7)     (2976, 7)       (8640, 7)       (2976, 7)
# print(self.seq_len)         # 96            96              96              96
# print(self.pred_len)        # 192           192             192             192
# print(len(self))            # 8353          2689            8353            2689

# #                           ETTm1-pretrain     ETTm1-test      ETTm2-pretrain     ETTm2-test
# print(self.data.shape)      # (69680, 7)    (69680, 7)      (69680, 7)      (69680, 7)
# print(self.data_x.shape)    # (34560, 7)    (11616, 7)      (34560, 7)      (11616, 7)
# print(self.data_y.shape)    # (34560, 7)    (11616, 7)      (34560, 7)      (11616, 7)
# print(self.seq_len)         # 96            96              96              96
# print(self.pred_len)        # 192           192             192             192
# print(len(self))            # 34273         11329           34273           11329

# # 对于train/test/vali具有相同长度的self.data，将其拆分为三段，pretrain:test:vali=7:2:1
# #                           ECL-pretrain       Traffic-pretrain   Exchange-pretrain  Weather-pretrain   ILI-pretrain
# print(self.data.shape)      # (26304, 321)  (17544, 862)    (7588, 8)       (52696, 21)     (966, 7)
# print(self.data_x.shape)    # (18412, 321)  (12280, 862)    (5311, 8)       (36887, 21)     (676, 7)
# print(self.data_y.shape)    # (18412, 321)  (12280, 862)    (5311, 8)       (36887, 21)     (676, 7)
# print(self.seq_len)         # 96            96              96              96              36
# print(self.pred_len)        # 192           192             192             192             48
# print(len(self))            # 18125         11993           5024            36600           593

# Pems                        TrainSet            ValiSet             TestSet
# print(self.data_x.shape)    # (15724, 358)      (5242, 358)         (5242, 358)
# print(self.data_y.shape)    # (15724, 358)      (5242, 358)         (5242, 358)

# Solar                       TrainSet            ValiSet             TestSet
# print(self.data_x.shape)    # (36792, 137)      (5352, 137)         (10608, 137)
# print(self.data_y.shape)    # (36792, 137)      (5352, 137)         (10608, 137)

Sunspot	            73924       Daily
SaugeenRiverFlow    23741	    Daily
USBirths            7305	    Daily
SolarPower	        7397222	    4Sec
WindPower	        7397147	    4Sec

ETTh1       从17420中截取前14400个时间点   all:17420   8640:2880:2880  
Pems        将全部时间点按照6:2:2划分
Solar       将全部时间点按照7:1:2划分
TSF_Long    将全部时间点按照6:2:2划分

转换为单变量后，各个数据集包含的单变量序列样本数量(sl=pl=96)
ETTh        ETTm        ECL         Traffic
121940      487760      8442621     15122928

Sunspot             Train           Val             Test
self.data:          (73924, 1)      (73924, 1)      (73924, 1)
self.data_x:        (44354, 1)      (14785, 1)      (14785, 1)
self.data_y:        (44354, 1)      (14785, 1)      (14785, 1)
SampleNum:          44163           14594           14594

SaugeenRiverFlow    Train           Val             Test
self.data:          (23741, 1)      (23741, 1)      (23741, 1)
self.data_x:        (14244, 1)      (4748, 1)       (4749, 1)
self.data_y:        (14244, 1)      (4748, 1)       (4749, 1)
SampleNum:          14053           4557            4558

USBirths            Train           Val             Test
self.data:          (7305, 1)       (7305, 1)       (7305, 1)
self.data_x:        (4383, 1)       (1461, 1)       (1461, 1)
self.data_y:        (4383, 1)       (1461, 1)       (1461, 1)
SampleNum:          4192            1270            1270

SolarPower          Train           Val             Test
self.data:          (7397222, 1)    (7397222, 1)    (7397222, 1)
self.data_x:        (4438333, 1)    (1479444, 1)    (1479445, 1)
self.data_y:        (4438333, 1)    (1479444, 1)    (1479445, 1)
SampleNum:          4438142         1479253         1479254

WindPower           Train           Val             Test
self.data:          (7397147, 1)    (7397147, 1)    (7397147, 1)
self.data_x:        (4438288, 1)    (1479429, 1)    (1479430, 1)
self.data_y:        (4438288, 1)    (1479429, 1)    (1479430, 1)
SampleNum:          4438097         1479238         1479239

SolarPower_DownSample
DownSampleScale:    1               2               5               10
self.data:          (7397222, 1)    (3698611, 1)    (1479444, 1)    (739722, 1)
SampleNum:          4438142         2218975         887475          443642

SolarPower_Step
StepScale:          1               12              24              48              96              192
SampleNum:          4438142         369846          184923          92462           46231           23116
"""


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, data_path, flag, size, downSampleScale=1, few_shot=1.0,
                 features='M', target='OT', scale=True, timeenc=0, freq='h'):

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.downSampleScale = downSampleScale
        self.__read_data__()

        self.few_shot = few_shot
        self.all_size = len(self.data) - self.seq_len - self.pred_len + 1
        self.few_size = int(self.few_shot * self.all_size)
        self.few_shot_idx = random.sample(range(0, self.all_size), self.few_size)

        self.scaler = StandardScaler()
        show_info = '{}-{}'.format(self.data_path.split('.')[0], flag)

        # print('')
        # print(show_info)            # ETTh1-pretrain   ETTh1-test   ETTh1-val   ETTh2-pretrain   ETTh2-test    ETTh2-val
        # print(self.data.shape)      # (8640, 7)     (3216, 7)    (3216, 7)   (8640, 7)     (3216, 7)     (3216, 7)
        # print(self.seq_len)         # 336
        # print(self.label_len)       # 0
        # print(self.pred_len)        # 96
        # print(len(self))            # 8209          2785         2785        8209          2785          2785

    def downSample(self):
        imp = (self.data.shape[0] // self.downSampleScale) * self.downSampleScale
        self.data = self.data[:imp].reshape(-1, self.downSampleScale, self.data.shape[-1]).mean(axis=1)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,                                              # Train
            12 * 30 * 24 - self.seq_len,                    # Val
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len       # Test
        ]
        border2s = [
            12 * 30 * 24,                                   # Train
            12 * 30 * 24 + 4 * 30 * 24,                     # Val
            12 * 30 * 24 + 8 * 30 * 24                      # Test
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # MultiVariate 2 MultiVariate
        # MultiVariate 2 SingleVariate
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        # SingleVariate 2 SingleVariate
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 从csv对象df_data中得到信号数据data
        # 若scale为True，则对Test和Val进行某种标准化处理，使得Test和Val与Train保持类似的分布特征
        # 若scale为False，则不对Test和Val进行处理
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data = data[border1:border2]

        # 对self.data做下采样
        if self.downSampleScale > 1:
            self.downSample()

    # 整条时间序列的长度为len(data)，按窗口长度为seq_len+pred_len划分若干个窗口，共得到(len(data) - (seq_len+pred_len) - 1)/1个窗口
    # 每个窗口中的前seq_len个时间戳作为lookback、后pred_len个时间戳作为forecast，其中lookback和forecast间重叠label_len个时间戳
    def __getitem__(self, index):
        idx = self.few_shot_idx[index]
        # Original
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return self.few_size

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, data_path, flag, size, downSampleScale=1, few_shot=1.0,
                 features='M', target='OT', scale=True, timeenc=0, freq='t'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.downSampleScale = downSampleScale
        self.__read_data__()

        self.few_shot = few_shot
        self.all_size = len(self.data) - self.seq_len - self.pred_len + 1
        self.few_size = int(self.few_shot * self.all_size)
        self.few_shot_idx = random.sample(range(0, self.all_size), self.few_size)

        self.scaler = StandardScaler()
        show_info = '{}-{}'.format(self.data_path.split('.')[0], flag)

        # print('')
        # print(show_info)            # ETTm1-pretrain   ETTm1-test   ETTm1-val    ETTm2-pretrain   ETTm2-test    ETTm2-val
        # print(self.data.shape)      # (34560, 7)    (11856, 7)   (11856, 7)   (34560, 7)    (11856, 7)    (11856, 7)
        # print(self.seq_len)         # 336
        # print(self.label_len)       # 0
        # print(self.pred_len)        # 96
        # print(len(self))            # 34129         11425        11425        34129         11425         11425

    def downSample(self):
        imp = (self.data.shape[0] // self.downSampleScale) * self.downSampleScale
        self.data = self.data[:imp].reshape(-1, self.downSampleScale, self.data.shape[-1]).mean(axis=1)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data = data[border1:border2]

        # 对self.data做下采样
        if self.downSampleScale > 1:
            self.downSample()

    # 整条时间序列的长度为len(data)，按窗口长度为seq_len+pred_len划分若干个窗口，共得到(len(data) - (seq_len+pred_len) - 1)/1个窗口
    # 每个窗口中的前seq_len个时间戳作为lookback、后label_len+pred_len个时间戳作为forecast，其中lookback和forecast间重叠label_len个时间戳
    def __getitem__(self, index):
        idx = self.few_shot_idx[index]
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return self.few_size

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path, flag, size, downSampleScale=1, few_shot=1.0,
                 features='M', target='OT', scale=True, timeenc=0, freq='h'):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.downSampleScale = downSampleScale
        self.__read_data__()

        self.few_shot = few_shot
        self.all_size = len(self.data) - self.seq_len - self.pred_len + 1
        self.few_size = int(self.few_shot * self.all_size)
        self.few_shot_idx = random.sample(range(0, self.all_size), self.few_size)

        self.scaler = StandardScaler()
        show_info = '{}-{}'.format(self.data_path.split('.')[0], flag)

        # print('')
        # print(show_info)            # ECL-pretrain     ECL-test     ECL-val       Traffic-pretrain    Traffic-test    Traffic-val    Exchange-pretrain   Exchange-test   Exchange-val    Weather-pretrain   Weather-test   Weather-val
        # print(self.data.shape)      # (18412, 321)  (4281, 321)  (4281, 321)   (12280, 862)     (2967, 862)     (2967, 862)    (5311, 8)        (1474, 8)       (1474, 8)       (36887, 21)     (8240, 21)     (8240, 21)
        # print(self.seq_len)         # 336
        # print(self.label_len)       # 0
        # print(self.pred_len)        # 96
        # print(len(self))            # 17981         3850         3850          11849            2536            2536           4880             1043            1043            36456           7809           7809

    def downSample(self):
        imp = (self.data.shape[0] // self.downSampleScale) * self.downSampleScale
        self.data = self.data[:imp].reshape(-1, self.downSampleScale).mean(axis=1)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data = data[border1:border2]

        # 对self.data做下采样
        if self.downSampleScale > 1:
            self.downSample()

    # 整条时间序列的长度为len(data)，按窗口长度为seq_len+pred_len划分若干个窗口，共得到(len(data) - (seq_len+pred_len) - 1)/1个窗口
    # 每个窗口中的前seq_len个时间戳作为lookback、后label_len+pred_len个时间戳作为forecast，其中lookback和forecast间重叠label_len个时间戳
    def __getitem__(self, index):
        idx = self.few_shot_idx[index]
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return self.few_size

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def ETT_Custom_Loader_tester():
    size = [336, 0, 96]

    for data in ['ETTh1', 'ETTh2']:
        root_path = './dataset/ETT-small/'
        data_path = '{}.csv'.format(data)
        for flag in ['train', 'test', 'val']:
            dataset = Dataset_ETT_hour(root_path=root_path, data_path=data_path, flag=flag, size=size)

    for data in ['ETTm1', 'ETTm2']:
        root_path = './dataset/ETT-small/'
        data_path = '{}.csv'.format(data)
        for flag in ['train', 'test', 'val']:
            dataset = Dataset_ETT_minute(root_path=root_path, data_path=data_path, flag=flag, size=size)

    root_path = './dataset/electricity/'
    data_path = 'electricity.csv'
    for flag in ['train', 'test', 'val']:
        dataset = Dataset_Custom(root_path=root_path, data_path=data_path, flag=flag, size=size)

    root_path = './dataset/traffic/'
    data_path = 'traffic.csv'
    for flag in ['train', 'test', 'val']:
        dataset = Dataset_Custom(root_path=root_path, data_path=data_path, flag=flag, size=size)

    root_path = './dataset/exchange_rate/'
    data_path = 'exchange_rate.csv'
    for flag in ['train', 'test', 'val']:
        dataset = Dataset_Custom(root_path=root_path, data_path=data_path, flag=flag, size=size)

    root_path = './dataset/weather/'
    data_path = 'weather.csv'
    for flag in ['train', 'test', 'val']:
        dataset = Dataset_Custom(root_path=root_path, data_path=data_path, flag=flag, size=size)


class Dataset_TSF_longUni(Dataset):
    def __init__(self, root_path, data_path, size, flag='train', downSampleScale=1.0, few_shot=1.0, step=1.0,):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.downSampleScale = downSampleScale
        self.step = step

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.few_shot = few_shot
        self.all_size = int((self.data.shape[0] - self.seq_len - self.pred_len) // self.step + 1)
        self.few_size = int(self.few_shot * self.all_size)
        self.few_shot_idx = random.sample(range(0, self.all_size), self.few_size)
        show_info = '{}-{}'.format(self.data_path.split('.')[0], flag)

        # print('')
        # print(show_info)            # Sunspot-pretrain   Sunspot-test   Sunspot-val    RiverFlow-pretrain    RiverFlow-test   RiverFlow-val   USBirths-pretrain   USBirths-test   USBirths-val   SolarPower-pretrain    SolarPower-test   SolarPower-val   WindPower-pretrain   WindPower-test   WindPower-val
        # print(self.data.shape)      # (59139, 1)      (7393, 1)      (7392, 1)      (18992, 1)         (2375, 1)        (2374, 1)       (5844, 1)        (731, 1)        (730, 1)       (5917777, 1)        (739723, 1)       (739722, 1)      (5917717, 1)      (739715, 1)      (739715, 1)
        # print(self.seq_len)         # 336
        # print(self.label_len)       # 0
        # print(self.pred_len)        # 96
        # print(len(self))            # 58708           6962           6961           18561              1944             1943            5413             300             299            5917346             739292             739291          5917286           739284           739284

    def downSample(self):
        imp = (self.data.shape[0] // self.downSampleScale) * self.downSampleScale
        self.data = self.data[:imp].reshape(-1, self.downSampleScale).mean(axis=1)

    def dropna(self, x):
        return x[~np.isnan(x)]

    def __read_data__(self):
        # 1、从.tsf文件中读取数据并去除其中的nan值
        df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
            os.path.join(
                self.root_path, self.data_path
            )
        )
        self.freq = frequency
        self.data = self.dropna(df.series_value[0]).astype(np.float32)

        # 2、对于SolaPower和WindPower这两个较大的数据集，可能需要需要做下采样
        if self.downSampleScale > 1:
            self.downSample()
        self.data = self.data.reshape(-1, 1)

        # 3、标准化
        train_ratio = 0.7
        test_ratio = 0.2
        train_data = self.data[:int(train_ratio * len(self.data)), :]
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(self.data)

        # 4、按照8:1:1的比例切分为train:val:test
        if self.flag == 'train':
            self.data = self.data[:int(train_ratio * len(self.data)), :]
        elif self.flag == 'test':
            self.data = self.data[int(train_ratio * len(self.data)): int((train_ratio + test_ratio) * len(self.data)), :]
        elif self.flag == 'val':
            self.data = self.data[int((train_ratio + test_ratio) * len(self.data)):, :]
        else:
            raise SystemExit('DATALOADER_TSF_LONG: flag is wrong!')

    def __getitem__(self, index):
        idx = self.few_shot_idx[index]
        # Step
        idx = int(idx * self.step)
        # Original
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return self.few_size


def TSFloader_long_tester():
    size = [336, 0, 96]

    # 1、展示所有单变量数据集的具体细节
    root_path = './dataset'
    data_path_ = ['Sunspot.tsf', 'SaugeenRiverFlow.tsf', 'USBirths.tsf', 'SolarPower.tsf', 'WindPower.tsf']
    for data_path in data_path_:
        for flag in ['train', 'test', 'val']:
            dataset = Dataset_TSF_longUni(root_path=root_path, data_path=data_path, size=size, flag=flag)

    # 2、对于超长数据集SolarPower和WindPower，可以使用【下采样】和【增大stride】以减小数据量
    root_path = './dataset'
    data_path = 'SolarPower.tsf'
    for down in [1, 10]:
        dataset = Dataset_TSF_longUni(root_path=root_path, data_path=data_path, size=size, flag='train', downSampleScale=down, step=1)
        # print(show_info)            # SolarPower-pretrain      SolarPower-pretrain
        # print(self.data.shape)      # (5917777, 1)          (591777, 1)
        # print(len(self))            # 5917346               591346
    for step in [1, 100]:
        dataset = Dataset_TSF_longUni(root_path=root_path, data_path=data_path, size=size, flag='train', downSampleScale=1, step=step)
        # print(show_info)            # SolarPower-pretrain      SolarPower-pretrain
        # print(self.data.shape)      # (5917777, 1)          (5917777, 1)
        # print(len(self))            # 5917346               59174


def getTimeSeriesDataset(data, input_len, pred_len, flag, down_scale=1.0, step=1.0):
    size = [input_len, 0, pred_len]

    if data in ['ETTh1', 'ETTh2']:
        root_path = './dataset/ETT-small/'
        data_path = '{}.csv'.format(data)
        dataset = Dataset_ETT_hour(root_path=root_path, data_path=data_path, flag=flag, size=size)
    elif data in ['ETTm1', 'ETTm2']:
        root_path = './dataset/ETT-small/'
        data_path = '{}.csv'.format(data)
        dataset = Dataset_ETT_minute(root_path=root_path, data_path=data_path, flag=flag, size=size)
    elif data in ['ECL']:
        root_path = './dataset/electricity/'
        data_path = 'electricity.csv'
        dataset = Dataset_Custom(root_path=root_path, data_path=data_path, flag=flag, size=size)
    elif data in ['Traffic']:
        root_path = './dataset/traffic/'
        data_path = 'traffic.csv'
        dataset = Dataset_Custom(root_path=root_path, data_path=data_path, flag=flag, size=size)
    elif data in ['Exchange']:
        root_path = './dataset/exchange_rate/'
        data_path = 'exchange_rate.csv'
        dataset = Dataset_Custom(root_path=root_path, data_path=data_path, flag=flag, size=size)
    elif data in ['Weather']:
        root_path = './dataset/weather/'
        data_path = 'weather.csv'
        dataset = Dataset_Custom(root_path=root_path, data_path=data_path, flag=flag, size=size)
    elif data in ['Sunspot']:
        root_path = './dataset'
        data_path = 'Sunspot.tsf'
        dataset = Dataset_TSF_longUni(root_path=root_path, data_path=data_path, size=size, flag=flag, downSampleScale=down_scale, step=step)
    elif data in ['RiverFlow']:
        root_path = './dataset'
        data_path = 'SaugeenRiverFlow.tsf'
        dataset = Dataset_TSF_longUni(root_path=root_path, data_path=data_path, size=size, flag=flag, downSampleScale=down_scale, step=step)
    elif data in ['USBirths']:
        root_path = './dataset'
        data_path = 'USBirths.tsf'
        dataset = Dataset_TSF_longUni(root_path=root_path, data_path=data_path, size=size, flag=flag, downSampleScale=down_scale, step=step)
    elif data in ['SolarPower']:
        root_path = './dataset'
        data_path = 'SolarPower.tsf'
        dataset = Dataset_TSF_longUni(root_path=root_path, data_path=data_path, size=size, flag=flag, downSampleScale=down_scale, step=step)
    elif data in ['WindPower']:
        root_path = './dataset'
        data_path = 'WindPower.tsf'
        dataset = Dataset_TSF_longUni(root_path=root_path, data_path=data_path, size=size, flag=flag, downSampleScale=down_scale, step=step)
    else:
        raise RuntimeError('DATALOADER: data is error!')

    print(f'DataLen of {data} is: {len(dataset)}')
    return dataset


if __name__ == '__main__':
    ETT_Custom_Loader_tester()
    TSFloader_long_tester()
