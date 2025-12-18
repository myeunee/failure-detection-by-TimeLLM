import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--debug_samples', type=int, default=0,
                    help='limit number of training samples for quick debug; 0 to disable')
# multi-task / evaluation extras (align with google_cluster_trace_repo)
parser.add_argument('--multi_task', action='store_true', default=False, help='enable multi-task (reg + cls)')
parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='classification loss weight')
parser.add_argument('--last_hour_eval', action='store_true', default=False, help='run last-window evaluation')
parser.add_argument('--last_hour_steps', type=int, default=12, help='steps for last-hour eval window')

# Trace dataset specific options
parser.add_argument('--reg_col', type=str, default='avg_usage_memory', help='regression target column for Trace dataset')
parser.add_argument('--cls_col', type=str, default='fail_in_window', help='classification target column for Trace dataset')
parser.add_argument('--trace_test_ratio', type=float, default=0.1, help='test set ratio for Trace dataset (instance-based split)')
parser.add_argument('--trace_val_ratio', type=float, default=0.1, help='validation set ratio for Trace dataset (instance-based split)')
parser.add_argument('--trace_split_file', type=str, default=None, help='path to JSON file storing instance splits for Trace dataset')
parser.add_argument('--trace_use_covariates', action='store_true', default=False, help='use additional covariates for Trace dataset')

# ablation options
parser.add_argument('--extra_head', type=str, default='none', choices=['none', 'mlp', 'lstm', 'mlp_lstm'],
                    help='optional extra module applied on patch embeddings before reprogramming')
parser.add_argument('--no_cleanup', action='store_true', default=True,
                    help='if set, do not delete checkpoints directory after training')
parser.add_argument('--use_deepspeed', action='store_true', default=False,
                    help='enable DeepSpeed only on CUDA; disabled by default (MPS/macOS not supported)')

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

# Configure accelerator
use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if use_mps:
    # avoid multiprocessing dataloader issues on macOS
    args.num_workers = 0

accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # Optional: limit all subsets for quick debug
    if args.debug_samples and args.debug_samples > 0:
        tN = min(args.debug_samples, len(train_data))
        vN = min(args.debug_samples // 4, len(vali_data))  # val은 1/4 크기
        teN = min(args.debug_samples // 4, len(test_data))  # test도 1/4 크기
        
        train_data = Subset(train_data, list(range(tN)))
        vali_data = Subset(vali_data, list(range(vN)))
        test_data = Subset(test_data, list(range(teN)))
        
        train_loader = DataLoader(
            train_data,
            batch_size=min(args.batch_size, tN),
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False
        )
        vali_loader = DataLoader(
            vali_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )
        test_loader = DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )
        accelerator.print(f"[Debug] Subset activated: train={tN}, val={vN}, test={teN}")

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    # When device_placement is False, we must place the model explicitly
    try:
        model.to(accelerator.device)
    except Exception:
        pass

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    accelerator.print(f"[INFO] Train dataset size: {len(train_data)}, Train steps per epoch: {train_steps}, Batch size: {args.batch_size}")
    accelerator.print(f"[INFO] Estimated time per epoch: {train_steps / 3.2 / 60:.1f} minutes (assuming 3.2 it/s)")
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)


    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder (Accelerator handles mixed precision automatically)
            with accelerator.autocast():
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                if args.multi_task and outputs.shape[-1] >= 2:
                    mem_pred, cls_pred = outputs[:, :, 0], outputs[:, :, 1]
                    mem_true, cls_true = batch_y[:, :, 0], batch_y[:, :, 1]
                    fail_ratio = cls_true.mean().item()
                    if fail_ratio == 0 or fail_ratio == 1:
                        pos_weight = torch.tensor([1.0], device=cls_pred.device)
                    else:
                        pos_weight = torch.tensor([(1 - fail_ratio) / fail_ratio], device=cls_pred.device)
                    loss_reg = criterion(mem_pred, mem_true)
                    loss_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(cls_pred, cls_true)
                    loss = loss_reg + args.cls_loss_weight * loss_cls
                else:
                    loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
                
                # Clear GPU cache every 100 iterations to prevent memory fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Accelerator handles gradient scaling automatically with mixed_precision
            accelerator.backward(loss)
            model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_out = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_out = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        vali_loss, vali_mae_loss = vali_out[0], vali_out[1]
        test_loss, test_mae_loss = test_out[0], test_out[1]
        if args.multi_task and len(vali_out) > 2 and vali_out[2] is not None:
            accelerator.print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE: {4:.7f} | V-ACC: {5:.4f} T-ACC: {6:.4f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss, vali_out[2], test_out[2]))
        else:
            accelerator.print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
if accelerator.is_local_main_process and not args.no_cleanup:
    path = './checkpoints'
    del_files(path)
    accelerator.print('success delete checkpoints')