'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import wandb
wandb.init(project = "Tensorrt-Re+Decouple_64", entity = "qqq1204")
import time
import glob
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.join('..', os.path.abspath(os.path.join(os.getcwd()))) )
from util_for_tensorrt import *
import torch
import utils
import logging
import argparse
import traceback
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# import flops_benchmark
from utils.visualizer import Visualizer
from config import Config
from lib import *
from utils import *

#------------------------
# evaluation metrics
#------------------------
from sklearn.decomposition import PCA
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt  # For graphics
import seaborn as sns
from torchvision.utils import save_image, make_grid

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Load Congfile.')
parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
parser.add_argument('--local-rank', type=int, default=0)
parser.add_argument('--nprocs', type=int, default=1)

parser.add_argument('--save_grid_image', action='store_true', help='Save samples?')
parser.add_argument('--save_output', action='store_true', help='Save logits?')
parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')
parser.add_argument('--resume', type=str, default='', help='resume model path.')

parser.add_argument('--distill-lamdb', type=float, default=0.0, help='initial distillation loss weight')

parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
parser.add_argument('--save', type=str, default='Checkpoints', help='experiment dir')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--weight', type=str, default='/home/minjae/ws/MotionRGBD/model.engine', help='random seed')

args = parser.parse_args()
args = Config(args)

try:
    if args.resume:
        args.save = os.path.split(args.resume)[0]
    else:
        args.save = '{}/{}-{}-{}-{}'.format(args.save, args.Network, args.dataset, args.type, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=[args.config] + glob.glob('./tools/*.py') + glob.glob('./lib/*'))
except:
    pass
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    return rt.mean().item()


def main(local_rank, nprocs, args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % local_rank)

    #---------------------------
    # Init distribution
    #---------------------------

    #----------------------------
    # build function
    #----------------------------
    
    engine, context, inputs_, outputs, bindings, stream = initialize(args.weight)

    # Dummy input data for testing
    input_data_a = np.random.rand(1, 3, 64, 224, 224).astype(np.float32)
    input_data_b = np.random.rand(1, 1, 64, 224, 224).astype(np.float32)

    # Set input shapes if dynamic
    input_binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    context.set_input_shape(input_binding_names[0], input_data_a.shape)
    context.set_input_shape(input_binding_names[1], input_data_b.shape)



    
    criterion = build_loss(args)

    valid_queue, valid_sampler = build_dataset(args, phase='valid')
    print(args)

    
    strat_epoch = 0
    args.epoch = strat_epoch

    if args.eval_only:
        valid_acc = infer(valid_queue, model, criterion, local_rank, strat_epoch)
        logging.info('valid_acc: {}'.format(valid_acc))
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if True:
        return tensor

@torch.no_grad()
def infer(valid_queue, model, criterion, local_rank, epoch):

    meter_dict = dict(
        Total_loss=AverageMeter(),
        CE_loss=AverageMeter(),
        Distil_loss=AverageMeter()
    )
    meter_dict.update(dict(
        cosin_similar=AverageMeter(),
    ))
    meter_dict.update(dict(
        Acc=AverageMeter()
    ))
    
    
    ## TensorRT
    engine, context, inputs_, outputs, bindings, stream = initialize(args.weight)

    # Dummy input data for testing
    input_data_a = np.random.rand(1, 3, 64, 224, 224).astype(np.float32)
    input_data_b = np.random.rand(1, 1, 64, 224, 224).astype(np.float32)

    # Set input shapes if dynamic
    input_binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    context.set_input_shape(input_binding_names[0], input_data_a.shape)
    context.set_input_shape(input_binding_names[1], input_data_b.shape)
    
    
    
    meter_dict['Infer_Time'] = AverageMeter()
    CE = criterion
    grounds, preds, v_paths = [], [], []
    output = {}
    for step, (inputs, heatmap, target, v_path) in enumerate(valid_queue):
        n = inputs.size(0)
        end = time.time()
        inputs, target, heatmap = map(lambda x: x.cuda(local_rank, non_blocking=True), [inputs, target, heatmap])
        
        ## Tensorrt
        inputs_np = inputs.cpu().numpy()
        heatmap_np = heatmap.cpu().numpy()
        preprocess_input(inputs_, inputs_np, heatmap_np)

        # Run inference
        do_inference(context, bindings, inputs_, outputs, stream)
        output_binding_name = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)][0]
        output_shape = context.get_tensor_shape(output_binding_name)
        logits = np.reshape(outputs[0][0], output_shape)
        
        
        meter_dict['Infer_Time'].update((time.time() - end) / n)
        

        globals()['Acc'] = calculate_accuracy2(logits, target)

        for name in meter_dict:
            if 'Acc' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))

        if step % args.report_freq == 0 and local_rank == 0:
            log_info = {
                'Epoch': epoch + 1,
                'Mini-Batch': '{:0>4d}/{:0>4d}'.format(step + 1, len(valid_queue.dataset) // (
                            args.test_batch_size * args.nprocs)),
            }
            log_info.update(dict((name, '{:.4f}'.format(value.avg)) for name, value in meter_dict.items()))
            print_func(log_info)


        if args.save_output:
            for t, logit in zip(v_path, logits):
                output[t] = logit
    #torch.distributed.barrier()
    grounds_gather = concat_all_gather(torch.tensor(grounds).cuda(local_rank))
    preds_gather = concat_all_gather(torch.tensor(preds).cuda(local_rank))
    grounds_gather, preds_gather = list(map(lambda x: x.cpu().numpy(), [grounds_gather, preds_gather]))

    if local_rank == 0:
        # v_paths = np.array(v_paths)[random.sample(list(wrong), 10)]
        v_paths = np.array(v_paths)
        grounds = np.array(grounds)
        preds = np.array(preds)
        wrong_idx = np.where(grounds != preds)
        v_paths = v_paths[wrong_idx[0]]
        grounds = grounds[wrong_idx[0]]
        preds = preds[wrong_idx[0]]
    return meter_dict['Acc'].avg

if __name__ == '__main__':
    wandb.config.update(args)
    if args.visdom['enable']:
        vis = Visualizer(args.visdom['visname'])
    try:
        
        torch.cuda.set_device(args.local_rank)
        main(args.local_rank, args.nprocs, args)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
            print('remove {}: Directory'.format(args.save))
            os.system('rm -rf {} \n mv {} ./Checkpoints/trash'.format(args.save, args.save))
        os._exit(0)
    except Exception:
        print(traceback.print_exc())
        if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
            print('remove {}: Directory'.format(args.save))
            os.system('rm -rf {} \n mv {} ./Checkpoints/trash'.format(args.save, args.save))
        os._exit(0)
    finally:
        torch.cuda.empty_cache()
        wandb.finish()
        
'''
command helper
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 tools/eval_tensorrt.py --config config/NvGesture_init.yml --nprocs 1 --eval_only

'''