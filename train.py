import builtins
import datetime
import argparse
from utils import get_logger
from torch import nn
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from utils import instantiation
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils import divide_train_val
from torch.optim import lr_scheduler
from evaluate import evaluate
from tqdm import tqdm
import torch.distributed as dist

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
    
def get_dataloader(align_type, config, device):
    train_indexes, val_indexes = divide_train_val(seed = config.data.divide_seed, align_type = align_type, train_ratio = config.data.train_ratio, device = device)
    train_indexes = {'train_indexes': train_indexes}
    val_indexes = {'val_indexes': val_indexes}
    
    train_dataset = instantiation(config.data.train, train_indexes)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = False, # must use False, see https://blog.csdn.net/Caesar6666/article/details/126893353
                                sampler = train_sampler,
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    
    val_dataset = instantiation(config.data.validation, val_indexes)
    val_dataloader = DataLoader(val_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = True, 
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    return train_sampler, train_dataloader, val_dataloader

def main():
    logger = get_logger('train')
    parser = argparse.ArgumentParser()
    # for distributed training
    parser.add_argument('--local_rank', default = -1, type = int)
    parser.add_argument('--dist_url', default = 'env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default = 1, type = int, help = 'number of distributed processes')
    parser.add_argument('--dist_on_itp', action = 'store_true')
    parser.add_argument('-t', '--type', help = 'align type(mtcnn or landmark), default: mtcnn', type = str, default = 'mtcnn')
    args = parser.parse_args()
    config_path = f'configs/train_{args.type}.yaml'
    
    init_distributed_mode(args)
    setup_for_distributed(get_rank() == 0)
    global device
    device = get_rank()
    
    # model
    cur_epoch = 0
    config = OmegaConf.load(config_path)
    model = instantiation(config.model)
    optim_state_dict = None
    if os.path.isfile(model.ckpt_path):
        cur_epoch, optim_state_dict = model.load_ckpt(model.ckpt_path, logger)
    model.train()
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, broadcast_buffers = False, find_unused_parameters = True)

    # data
    train_sampler, train_dataloader, val_dataloader = get_dataloader(args.type, config, device)
    
    # criterion
    criterion = instantiation(config.loss)
    
    # optimizer
    params = list(model.module.parameters())
    optimizer = torch.optim.AdamW(params = params, lr = config.optimizer.max_lr)
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
    lambda_lr = lambda epoch: ((config.optimizer.base_lr - config.optimizer.max_lr) / config.optimizer.linear_epochs * epoch + config.optimizer.max_lr) / config.optimizer.max_lr if epoch < config.optimizer.linear_epochs else (config.optimizer.min_lr + 0.5 * (config.optimizer.base_lr - config.optimizer.min_lr) * (1 + np.cos((epoch - config.optimizer.linear_epochs) / (config.optimizer.max_epochs - config.optimizer.linear_epochs) * np.pi))) / config.optimizer.max_lr
    # lambda_lr = lambda epoch: ((config.optimizer.min_lr + 0.5 * (config.optimizer.base_lr - config.optimizer.min_lr) * (1 + np.cos(epoch / (config.optimizer.max_epochs) * np.pi))) / config.optimizer.base_lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_lr, last_epoch=cur_epoch - 1)
    # start loop
    best_accuracy = 0
    accuracy_flag = 0
    counter_flag = 0
    try:
        for epoch in range(cur_epoch, config.optimizer.max_epochs):
            train_sampler.set_epoch(epoch)
            # train
            model.train()
            epoch_loss, Flag = train(optimizer, scheduler, epoch, model, criterion, train_dataloader, logger)
            if dist.get_rank() == 0 and (epoch + 1) % config.val_interval == 0:
                # validation
                model.eval()
                accuracy, _ = evaluate(model, val_dataloader, logger, device)
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
            else:
                logger.info('Epoch: {}, Loss: {:.4f}, lr: {:.4f}'.format(epoch, epoch_loss, optimizer.param_groups[0]['lr']))

            if Flag:
                counter_flag += 1
                logger.info('counter_flag: {}'.format(counter_flag))
            else:
                counter_flag = 0
            if counter_flag >= 5:
                break
    except Exception as e:
        logger.error(e)
    finally:
        if dist.get_rank() == 0:
            logger.info('Final Best Accuracy: {:.4f}'.format(best_accuracy))
            model.module.save_ckpt(model.module.save_ckpt_path, -1, best_accuracy, optimizer, logger, start_time)
    
def train(optimizer, scheduler, epoch, model, criterion, train_dataloader, logger):
    # loss of an epoch
    ave_loss = 0
    counter = 0
    Flag = False
    tqdm_iter = tqdm(train_dataloader, leave=False)
    for batch in tqdm_iter:
        anc_img, pos_img, neg_img = batch['anc'].to(device), batch['pos'].to(device), batch['neg'].to(device)
        anc_fea, pos_fea, neg_fea = model(anc_img), model(pos_img), model(neg_img)
        
        loss, count = criterion(anc_fea, pos_fea, neg_fea)
        
        ave_loss += loss.item() * count
        counter += count
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm_iter.set_description('Epoch: {}, Loss: {:.4f}, lr: {:.6f}, Count: {}'.format(epoch, loss, optimizer.param_groups[0]['lr'], count))
    
    scheduler.step()
    logger.info('counter: {}'.format(counter))
    if counter < 20:
        Flag = True
    return (ave_loss / counter, Flag) if counter != 0 else (0, Flag)
    
    
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m-%d_%H-%M-%S")
    
    main()
    