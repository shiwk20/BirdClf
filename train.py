import builtins
import datetime
import argparse
from utils import get_logger, instantiation, set_seed
from torch import nn
import torch
import os
from torch.cuda.amp import autocast
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from evaluate import evaluate
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
    parser = argparse.ArgumentParser()
    # for distributed training
    parser.add_argument('--local_rank', default = -1, type = int)
    parser.add_argument('--dist_url', default = 'env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default = 1, type = int, help = 'number of distributed processes')
    parser.add_argument('--dist_on_itp', action = 'store_true')
    parser.add_argument('-c', '--config', help = 'config files containing all configs', type = str, default = '')
    args = parser.parse_args()
    config_path = args.config
    logger = get_logger(os.path.basename(config_path).split('.')[0])
    
    init_distributed_mode(args)
    setup_for_distributed(get_rank() == 0)
    global device
    device = get_rank()
    
    # model
    cur_epoch = 0
    config = OmegaConf.load(config_path)
    set_seed(config.random_seed)
    
    model = instantiation(config.model)
    optim_state_dict = None
    if os.path.isfile(model.ckpt_path):
        cur_epoch, optim_state_dict = model.load_ckpt(model.ckpt_path, logger)
    model.train()
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, broadcast_buffers = False, find_unused_parameters = True)

    # data
    train_dataset = TrainDataset(config.data.img_size, config.data.data_path, config.data.augment)
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
    optimizer = torch.optim.AdamW(params = params, lr = config.optimizer.lr, weight_decay = config.optimizer.weight_decay)
    
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True) 
    
    # start loop
    best_accuracy = 0
    accuracy_flag = 0
    epoch_loss = 0
    accuracy = 0
    # try:
    for epoch in range(cur_epoch, config.max_epochs):
        train_sampler.set_epoch(epoch)
        # train
        model.train()
        epoch_loss = train(optimizer, scheduler, epoch, model, criterion, train_dataloader, logger)
        
        logger.info('Epoch: {}, Loss: {:.4f}, lr: {:.4f}'.format(epoch, epoch_loss, optimizer.param_groups[0]['lr']))
        
        if dist.get_rank() == 0 and (epoch + 1) % config.val_interval == 0:
            # validation
            model.eval()
            accuracy = evaluate(model, val_dataloader, logger, device)
            if accuracy > best_accuracy:
                accuracy_flag = 0
                best_accuracy = accuracy
                # save ckpt
                model.module.save_ckpt(model.module.save_ckpt_path, epoch + 1, best_accuracy, optimizer, logger, start_time)
            else:
                accuracy_flag += 1
                logger.info('accuracy_flag: {}'.format(accuracy_flag))
                if accuracy_flag >= 5:
                    break
            logger.info('Epoch: {}, Loss: {:.4f}, lr: {:.4f}, Accuracy: {:.4f}, Best Accuracy: {:.4f}'.format(epoch, epoch_loss, optimizer.param_groups[0]['lr'], accuracy, best_accuracy))

    # except Exception as e:
    #     logger.error(e)
    # finally:
    #     if dist.get_rank() == 0:
    #         logger.info('Final epoch: {}, Loss: {:.4f}, lr: {:.4f}, Accuracy: {:.4f}, Best Accuracy: {:.4f}'.format(epoch, epoch_loss, optimizer.param_groups[0]['lr'], accuracy, best_accuracy))
    #         model.module.save_ckpt(model.module.save_ckpt_path, epoch + 1, accuracy, optimizer, logger, start_time, is_final = True)
    
def train(optimizer, scheduler, epoch, model, criterion, train_dataloader, logger):
    # loss of an epoch
    count = 0
    ave_loss = 0
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
        tqdm_iter.set_description('Epoch: {}, Loss: {:.4f}, lr: {:.6f}'.format(epoch, loss, optimizer.param_groups[0]['lr']))
        
    
    scheduler.step(epoch)
    return ave_loss / count
    
    
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m-%d_%H-%M-%S")
    
    main()
    