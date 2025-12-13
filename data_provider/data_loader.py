import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
import json

warnings.filterwarnings('ignore')


# ---------------- Trace Multi-Task Dataset -----------------
class Dataset_Trace(Dataset):
    """Trace dataset (Google cluster instances) for multi-task forecasting.

    Assumptions:
    - CSV has columns: collection_id, instance_index, avg_usage_memory, fail_in_window plus covariates.
    - Each row is a 5-min window ordered by original file order.
    - We only use two variables (regression + classification) for now to keep enc_in=2.
    - Instance-based splitting: select a ratio of instances for test and validation; rest for training.
    - Scaling: only apply StandardScaler to regression column; classification kept as raw 0/1.
    """
    scaler_mean = None
    scaler_std = None
    instance_splits = None  # {'train': set(keys), 'val': set(keys), 'test': set(keys)}

    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='trace.csv',
                 reg_col='avg_usage_memory', cls_col='fail_in_window',
                 target=None, scale=True, timeenc=0, freq='t', percent=100,
                 trace_test_ratio=0.1, trace_val_ratio=0.1, seed=2021,
                 trace_split_file=None,
                 use_covariates=False,
                 seasonal_patterns=None):
        if size is None:
            self.seq_len = 512
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        self.set_type = flag
        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.reg_col = reg_col
        self.cls_col = cls_col
        self.trace_test_ratio = trace_test_ratio
        self.trace_val_ratio = trace_val_ratio
        self.seed = seed
        self.trace_split_file = trace_split_file
        self.use_covariates = use_covariates
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        # cache per-instance last window indices for custom eval
        self._compute_instance_ranges()

    def _build_instance_splits(self, df):
        keys = df[['collection_id', 'instance_index']].astype(str).agg('__'.join, axis=1)
        unique_keys = keys.unique().tolist()
        # If split file specified, try to load
        if self.trace_split_file and os.path.exists(self.trace_split_file):
            with open(self.trace_split_file, 'r') as f:
                loaded = json.load(f)
            Dataset_Trace.instance_splits = {
                'train': set(loaded.get('train', [])),
                'val': set(loaded.get('val', [])),
                'test': set(loaded.get('test', [])),
            }
            return
        # Else create new split
        rng = np.random.default_rng(self.seed)
        rng.shuffle(unique_keys)
        n_total = len(unique_keys)
        n_test = max(1, int(n_total * self.trace_test_ratio))
        n_val = max(1, int(n_total * self.trace_val_ratio))
        test_keys = set(unique_keys[:n_test])
        val_keys = set(unique_keys[n_test:n_test + n_val])
        train_keys = set(unique_keys[n_test + n_val:])
        Dataset_Trace.instance_splits = {
            'train': train_keys,
            'val': val_keys,
            'test': test_keys
        }
        # Persist split if requested
        if self.trace_split_file:
            with open(self.trace_split_file, 'w') as f:
                json.dump({k: sorted(list(v)) for k, v in Dataset_Trace.instance_splits.items()}, f, indent=2)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Initialize splits once
        if Dataset_Trace.instance_splits is None:
            self._build_instance_splits(df_raw)
        keys_series = df_raw[['collection_id', 'instance_index']].astype(str).agg('__'.join, axis=1)
        mask = keys_series.apply(lambda k: k in Dataset_Trace.instance_splits[self.set_type])
        df_subset = df_raw[mask].reset_index(drop=True)
        # keep key series for per-instance ops
        self.key_series = df_subset[['collection_id','instance_index']].astype(str).agg('__'.join, axis=1).values

        # Optionally apply percent (subset shortening for all splits)
        if self.percent < 100:
            cut = int(len(df_subset) * self.percent / 100)
            df_subset = df_subset.iloc[:cut]

        if self.reg_col not in df_subset.columns or self.cls_col not in df_subset.columns:
            raise ValueError(f"Trace dataset missing required columns '{self.reg_col}' or '{self.cls_col}'.")
        # Build channels: [regression, classification, covariates...]
        data_reg = df_subset[self.reg_col].astype(np.float32).values.reshape(-1, 1)
        data_cls = df_subset[self.cls_col].astype(np.float32).values.reshape(-1, 1)  # keep 0/1
        # Fit scaler on regression column (train only)
        if Dataset_Trace.scaler_mean is None and self.set_type == 'train' and self.scale:
            Dataset_Trace.scaler_mean = data_reg.mean()
            Dataset_Trace.scaler_std = data_reg.std() + 1e-8
        if self.scale and Dataset_Trace.scaler_mean is not None:
            data_reg = (data_reg - Dataset_Trace.scaler_mean) / Dataset_Trace.scaler_std
        # Include all other numeric covariates (excluding keys and targets) if enabled
        if self.use_covariates:
            exclude = {'collection_id','instance_index', self.reg_col, self.cls_col}
            cov_cols = [c for c in df_subset.columns if c not in exclude]
            cov_df = df_subset[cov_cols]
            cov_numeric = cov_df.select_dtypes(include=[np.number]).astype(np.float32).values if len(cov_cols)>0 else np.zeros((len(df_subset),0),dtype=np.float32)
        else:
            cov_numeric = np.zeros((len(df_subset),0), dtype=np.float32)
        data_all = np.concatenate([data_reg, data_cls, cov_numeric], axis=1).astype(np.float32)

        # time features: simple positional/time encoding using index if no timestamp
        if 'date' in df_subset.columns:
            df_stamp = df_subset[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values.astype(np.float32)
            else:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0).astype(np.float32)
        else:
            # Fallback: simple positional index
            idx = np.arange(len(df_subset), dtype=np.float32).reshape(-1, 1)
            data_stamp = idx

        self.data_x = data_all
        self.data_y = data_all
        self.data_stamp = data_stamp

    def _compute_instance_ranges(self):
        # compute start/end (exclusive) indices per instance within this split
        keys = self.key_series
        self.instance_last_start_indices = []
        if len(keys) == 0:
            return
        # find boundaries where key changes
        boundaries = np.where(np.concatenate([[True], keys[1:] != keys[:-1], [True]]))[0]
        # pairs of (start, end)
        ranges = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
        for s, e in ranges:
            length = e - s
            # s_begin for last window so that r_end aligns with e
            s_begin = e - self.pred_len - self.seq_len
            if s_begin >= s and s_begin >= 0:
                self.instance_last_start_indices.append(s_begin)

    def get_last_window_indices(self):
        return self.instance_last_start_indices

    def __getitem__(self, index):
        # Single multivariate sequence windows
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform_reg(self, data):
        if Dataset_Trace.scaler_mean is None:
            return data
        return (data * Dataset_Trace.scaler_std) + Dataset_Trace.scaler_mean


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
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

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

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

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
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

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

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

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
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
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        or for trace.csv: no date column, use index-based time encoding
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        
        # date 컬럼이 없으면 인덱스 기반으로 생성 (trace.csv용)
        if 'date' not in df_raw.columns:
            # 인덱스를 datetime으로 변환 (단순히 순서만 표현)
            df_raw['date'] = pd.date_range(start='2020-01-01', periods=len(df_raw), freq='5min')
            # elapsed_time_seconds는 feature로 유지
        else:
            cols.remove('date')
        
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

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

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

