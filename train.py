import builtins
import datetime
import argparse
from utils import get_logger, instantiation, set_seed
from torch import nn
import torch
import shutil
import warnings
warnings.filterwarnings("ignore")
import os
from torch.cuda.amp import autocast
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from test import evaluate, plot_curve
from tqdm import tqdm
import torch.distributed as dist
from dataset import TrainDataset, ValDataset

def setup_for_distributed(is_master):
    """
    disable printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def main():
    # model
    model = instantiation(config.model)
    
    cur_epoch = 0
    train_accs = []
    val_accs = []
    train_losses = []
    lrs = []
    optim_state_dict = None
    
    if config.resume:
        cur_epoch, optim_state_dict, train_accs, val_accs, train_losses, lrs = model.load_ckpt(config.ckpt_path, logger)
        
    model.train()
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, broadcast_buffers = False, find_unused_parameters = False)

    # data
    train_dataset = TrainDataset(config.data.img_size, config.data.data_path, config.data.augment, resized_crop=config.data.resized_crop, herizon_flip= config.data.herizon_flip, vertical_flip = config.data.vertical_flip, random_affine=config.data.random_affine)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = False, # must use False, see https://blog.csdn.net/Caesar6666/article/details/126893353
                                sampler = train_sampler,
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    
    val_dataset = ValDataset(config.data.img_size, config.data.data_path)
    val_dataloader = DataLoader(val_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = True, 
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    
    # criterion
    criterion = instantiation(config.loss)
    
    # optimizer
    params = list(model.module.parameters())
    if config.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(params = params, lr = config.optimizer.lr, weight_decay = config.optimizer.weight_decay)
    elif config.optimizer.type == 'SGD':
        optimizer = torch.optim.SGD(params = params, lr = config.optimizer.lr, momentum = config.optimizer.momentum, weight_decay = config.optimizer.weight_decay)
    elif config.optimizer.type == 'Adam':
        optimizer = torch.optim.Adam(params = params, lr = config.optimizer.lr, weight_decay = config.optimizer.weight_decay)
        
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
    
    if config.scheduler.type == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = config.scheduler.mode, factor = config.scheduler.factor, patience=config.scheduler.patience, verbose=config.scheduler.verbose)
    elif config.scheduler.type == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.scheduler.T_max, eta_min = config.scheduler.eta_min)
    elif config.scheduler.type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size = config.scheduler.step_size, gamma = config.scheduler.gamma)
    
    # start loop
    best_accuracy = 0 if len(val_accs) == 0 else max(val_accs)
    accuracy_flag = 0
    try:
        for epoch in range(cur_epoch, config.max_epochs):
            train_sampler.set_epoch(epoch)
            # train
            model.train()
            train_losses.append(train(optimizer, scheduler, epoch, model, criterion, train_dataloader))
            lrs.append(optimizer.param_groups[0]['lr'])

            logger.info('Epoch: {}, Loss: {:.6f}, lr: {:.8f}'.format(epoch, train_losses[-1], lrs[-1]))

            if dist.get_rank() == 0 and (epoch + 1) % config.val_interval == 0:
                # validation
                model.eval()
                train_accs.append(evaluate(model, train_dataloader, logger, device, type = 'train'))
                val_accs.append(evaluate(model, val_dataloader, logger, device, type = 'val'))
                if val_accs[-1] > best_accuracy:
                    accuracy_flag = 0
                    best_accuracy = val_accs[-1]
                    # save ckpt
                    model.module.save_ckpt(save_ckpt_path, epoch, train_accs, val_accs, train_losses, lrs, optimizer, logger)
                else:
                    accuracy_flag += 1
                    logger.info('accuracy_flag: {}'.format(accuracy_flag))
                    if accuracy_flag >= config.accuracy_thre:
                        break
    except Exception as e:
        logger.error('Error:' + e)       
    finally:    
        # save final ckpt
        if dist.get_rank() == 0:
            model.module.save_ckpt(save_ckpt_path, epoch, train_accs, val_accs, train_losses, lrs, optimizer, logger)
            logger.info('Training finished! Training time: {}'.format(datetime.datetime.now() - start_time))
            plot_curve(config.val_interval, epoch, train_accs, val_accs, train_losses, lrs, res_path)
    
    
def train(optimizer, scheduler, epoch, model, criterion, train_dataloader):
    # average loss of an epoch
    ave_loss = 0
    count = 0
    tqdm_iter = tqdm(train_dataloader, leave=False)
    for idx, (imgs, labels) in enumerate(tqdm_iter):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        count += len(imgs)
        ave_loss += loss.item() * len(imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm_iter.set_description('Epoch: {}, Loss: {:.6f}, lr: {:.8f}'.format(epoch, loss, optimizer.param_groups[0]['lr']))
        
    scheduler.step(epoch)
    return ave_loss / count
    
    
if __name__ == '__main__':
    # 训练前的准备
    res_path = 'res'
    parser = argparse.ArgumentParser()
    # for distributed training
    parser.add_argument('--local_rank', default = -1, type = int)
    parser.add_argument('--dist_url', default = 'env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default = 1, type = int, help = 'number of distributed processes')
    parser.add_argument('--dist_on_itp', action = 'store_true')
    parser.add_argument('-c', '--config', help = 'config files containing all configs', type = str, default = '')
    args = parser.parse_args()
    
    config_path = args.config
    config = OmegaConf.load(config_path)
    set_seed(config.random_seed)
    
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%m-%d_%H-%M-%S")
    config_type = os.path.basename(config_path).split('.')[0]
    res_path = os.path.join(res_path, config_type + '_' + start_time_str)
    has_dir = False
    
    if config.resume:
        if os.path.isfile(config.ckpt_path):
            if config.ckpt_path.find('train') != -1:
                res_path = os.path.dirname(os.path.dirname(config.ckpt_path))
                has_dir = True
        else:
            print(f'=> set to resume, but {config.ckpt_path} is not a file')
            exit(1)
    if not has_dir:
        os.makedirs(res_path, exist_ok=True)
        shutil.copyfile(config_path, os.path.join(res_path, 'config.yaml'))
    
    log_path = os.path.join(res_path, 'logs')
    save_ckpt_path = os.path.join(res_path, 'ckpts')
    
    logger = get_logger(log_path, type = 'train')
    
    logger.info(f'res_path: {res_path}')
    logger.info('config: {}'.format(config))
    
    init_distributed_mode(args)
    setup_for_distributed(dist.get_rank() == 0)
    device = get_rank()
    
    # 训练的主函数
    main()
    