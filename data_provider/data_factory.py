from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, Dataset_Trace
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'Trace': Dataset_Trace,
    'm4': Dataset_M4,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.eval_batch_size  # Use eval_batch_size for test
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = True
        batch_size = args.eval_batch_size  # Use eval_batch_size for val
        freq = args.freq
    else:  # train
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    elif args.data == 'Trace':
        # Trace dataset with multi-task support
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            reg_col=getattr(args, 'reg_col', 'avg_usage_memory'),
            cls_col=getattr(args, 'cls_col', 'fail_in_window'),
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            trace_test_ratio=getattr(args, 'trace_test_ratio', 0.1),
            trace_val_ratio=getattr(args, 'trace_val_ratio', 0.1),
            seed=getattr(args, 'seed', 2021),
            trace_split_file=getattr(args, 'trace_split_file', None),
            use_covariates=getattr(args, 'trace_use_covariates', False),
            scale=True,
            seasonal_patterns=None
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
