import sys
import os
# sys.stdout = open('test0.txt', 'w')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import numpy as np
import random
import copy
import time
import math
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.models import create_model, list_models
from timm.data import Mixup
from timm.utils import accuracy, AverageMeter
import models
from DataSelect import *
from DataSplit import *
from dependency_criterion import *


# python testsmall.py --model deit_small_patch16_224 --datatrain_path ../TinyImageNet/tiny-imagenet-200 --dataval_path ../TinyImageNet/tiny-imagenet-200 --batch_size 64 --neuron_pruning --head_pruning --seed 2025 --idx 0
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')    
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL', help='Name of model')
    parser.add_argument('--datatrain_path', default='../data', type=str, help='datatrain path')
    parser.add_argument('--dataval_path', default='../data', type=str, help='dataval path')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--neuron_pruning', action='store_true', default=False)
    parser.add_argument('--head_pruning', action='store_true', default=False)
    parser.add_argument('--token_pruning', action='store_true', default=False)
    parser.add_argument('--neuron_sparsity', type=float, default=0.)
    parser.add_argument('--head_sparsity', type=float, default=0.)
    parser.add_argument('--token_sparsity', type=float, default=0.) 

    return parser


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):  
    # 设置随机种子和编号
    set_seed(args.seed)
    idx = args.idx
    sys.stdout = open('Small'+str(idx)+'.txt', 'w')

    # 选择数据集  
    print(f"Loading data")
    worker_init_fn = lambda worker_id: (np.random.seed(args.seed+worker_id), random.seed(args.seed+worker_id))
    class_indices_val = [i for i in range(int(idx*20), int((idx+1)*20))]
    # class_indices_val = [i for i in range(0, 200)]
    data_loader_val = load_imagenet_subset(args.dataval_path, class_indices_val, args.batch_size, 
                                           args.num_workers, worker_init_fn, train=False)
    
    train = load_imagenet(args.datatrain_path, args.batch_size, args.num_workers, worker_init_fn, train=True)
    E = EdgeDevice(data_loader_val, args.batch_size, args.num_workers, worker_init_fn)
    E.Calculate_data_distribution()
    E.set_selected_dataset(train)
    data_loader_train = E.selected_dataset
    data_finetune = E.selected_dataset_finetune

    # save_data_loaders(data_loader_train, data_finetune, 
    #                 file_path_train='data_loader_train'+str(idx)+'.pth', 
    #                 file_path_finetune='data_finetune'+str(idx)+'.pth')

    # data_loader_train, data_finetune = load_data_loaders(
    # file_path_train='data_loader_train'+str(idx)+'.pth', 
    # file_path_finetune='data_finetune'+str(idx)+'.pth' , 
    # batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    # data_loader_train = load_imagenet_random(args.datatrain_path, args.batch_size*10, 
    #                                              args.batch_size, args.num_workers, worker_init_fn, train=True)
    # data_finetune = load_imagenet_random(args.datatrain_path, args.batch_size*100, 
    #                                          args.batch_size, args.num_workers, worker_init_fn, train=True)

    # 选择模型
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,         # 要使用的模型的名称或类型
        pretrained=False,   # 表示不使用预训练的权重
        num_classes=200,   # 模型的输出类别数
    )
    # 加载预训练模型
    model.load_state_dict(torch.load("../logs/deit_small.pth"))
    
    # 检查是否有可用的 GPU    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 开启 GPU 加速设置
    torch.backends.cudnn.benchmark = True

    # 将模型移动到设备
    model.to(device)
    # 前向传播以确保自定义模块初始化
    with torch.no_grad():
        # 创建虚拟输入, 假设输入是标准的图像大小 (batch_size, channels, height, width)
        dummy_input = torch.randn(args.batch_size, 3, 224, 224).to(device)
        output = model(dummy_input)
    # 如果使用多 GPU 将模型封装为 DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 模型微调
    print("Fine-tuning model")
    top1 = gp_evaluate(data_loader_val, model, device)
    print(top1)

    print("Neuron rank")
    neuron_rank = mlp_neuron_rank(model, data_loader_train, device) 
    print("Head rank")
    head_rank = attn_head_rank(model, data_loader_train, device)

    # save_ranks(neuron_rank, head_rank, 
    #            neuron_rank_path='neuron_rank'+str(idx)+'.pth', 
    #            head_rank_path='head_rank'+str(idx)+'.pth')

    print("Layer rank0")
    _, n_layer_rank0, h_layer_rank0= layer_rank(model, data_loader_train, device, 0.9, neuron_rank, head_rank)
    print("\nNormalized Neuron Layer rank0")
    norm_neuron_rank00 = normalized_layer_rank(n_layer_rank0, alpha=1.1)
    print("\nNormalized Head Layer rank0")
    norm_head_rank00 = normalized_layer_rank(h_layer_rank0, alpha=1.1)
    print("\nLayer ratio0")
    ratio00 = rank_to_ratio(norm_neuron_rank00, 0.886) + rank_to_ratio(norm_head_rank00, 0.928)


    print("Layer rank1")
    all_layer_rank1, n_layer_rank1, h_layer_rank1= layer_rank(model, data_loader_train, device, 0.8, neuron_rank, head_rank)
    # print("\nNormalized Layer rank1")
    # norm_rank1 = normalized_layer_rank(all_layer_rank1, alpha=1)
    print("\nNormalized Neuron Layer rank1")
    norm_neuron_rank10 = normalized_layer_rank(n_layer_rank1, alpha=1.1)
    # norm_neuron_rank11 = normalized_layer_rank(n_layer_rank1, alpha=1.2)
    # norm_neuron_rank12 = normalized_layer_rank(n_layer_rank1, alpha=1.3)
    print("\nNormalized Head Layer rank1")
    norm_head_rank10 = normalized_layer_rank(h_layer_rank1, alpha=1.1)
    # norm_head_rank11 = normalized_layer_rank(h_layer_rank1, alpha=1.2)
    # norm_head_rank12 = normalized_layer_rank(h_layer_rank1, alpha=1.3)
    print("\nLayer ratio1")
    ratio10 = rank_to_ratio(norm_neuron_rank10, 0.772) + rank_to_ratio(norm_head_rank10, 0.856)
    # ratio11 = rank_to_ratio(norm_neuron_rank11, 0.772) + rank_to_ratio(norm_head_rank11, 0.856)
    # ratio12 = rank_to_ratio(norm_neuron_rank12, 0.772) + rank_to_ratio(norm_head_rank12, 0.856)


    print("Layer rank2")
    all_layer_rank2, n_layer_rank2, h_layer_rank2= layer_rank(model, data_loader_train, device, 0.7, neuron_rank, head_rank)
    # print("\nNormalized Layer rank2")
    # norm_rank2 = normalized_layer_rank(all_layer_rank2, alpha=1)
    print("\nNormalized Neuron Layer rank2")
    norm_neuron_rank20 = normalized_layer_rank(n_layer_rank2, alpha=1.1)
    # norm_neuron_rank21 = normalized_layer_rank(n_layer_rank2, alpha=1.2)
    # norm_neuron_rank22 = normalized_layer_rank(n_layer_rank2, alpha=1.3)
    print("\nNormalized Head Layer rank2")
    norm_head_rank20 = normalized_layer_rank(h_layer_rank2, alpha=1.1)
    # norm_head_rank21 = normalized_layer_rank(h_layer_rank2, alpha=1.2)
    # norm_head_rank22 = normalized_layer_rank(h_layer_rank2, alpha=1.3)
    print("\nLayer ratio2")
    ratio20 = rank_to_ratio(norm_neuron_rank20, 0.658) + rank_to_ratio(norm_head_rank20, 0.784)
    # ratio21 = rank_to_ratio(norm_neuron_rank21, 0.658) + rank_to_ratio(norm_head_rank21, 0.784)
    # ratio22 = rank_to_ratio(norm_neuron_rank22, 0.658) + rank_to_ratio(norm_head_rank22, 0.784)


    print("Layer rank3")
    all_layer_rank3, n_layer_rank3, h_layer_rank3= layer_rank(model, data_loader_train, device, 0.6, neuron_rank, head_rank)
    # print("\nNormalized Layer rank3")
    # norm_rank3 = normalized_layer_rank(all_layer_rank3, alpha=1)
    print("\nNormalized Neuron Layer rank3")
    norm_neuron_rank30 = normalized_layer_rank(n_layer_rank3, alpha=1.1)
    # norm_neuron_rank31 = normalized_layer_rank(n_layer_rank3, alpha=1.2)
    # norm_neuron_rank32 = normalized_layer_rank(n_layer_rank3, alpha=1.3)
    print("\nNormalized Head Layer rank3")
    norm_head_rank30 = normalized_layer_rank(h_layer_rank3, alpha=1.1)
    # norm_head_rank31 = normalized_layer_rank(h_layer_rank3, alpha=1.2)
    # norm_head_rank32 = normalized_layer_rank(h_layer_rank3, alpha=1.3)
    print("\nLayer ratio3")
    ratio30 = rank_to_ratio(norm_neuron_rank30, 0.544) + rank_to_ratio(norm_head_rank30, 0.712)
    # ratio31 = rank_to_ratio(norm_neuron_rank31, 0.544) + rank_to_ratio(norm_head_rank31, 0.712)
    # ratio32 = rank_to_ratio(norm_neuron_rank32, 0.544) + rank_to_ratio(norm_head_rank32, 0.712)


    print("Layer rank4")
    all_layer_rank4, n_layer_rank4, h_layer_rank4= layer_rank(model, data_loader_train, device, 0.5, neuron_rank, head_rank)
    # print("\nNormalized Layer rank4")
    # norm_rank4 = normalized_layer_rank(all_layer_rank4, alpha=1)
    print("\nNormalized Neuron Layer rank4")
    norm_neuron_rank40 = normalized_layer_rank(n_layer_rank4, alpha=1.1)
    # norm_neuron_rank41 = normalized_layer_rank(n_layer_rank4, alpha=1.2)
    # norm_neuron_rank42 = normalized_layer_rank(n_layer_rank4, alpha=1.3)
    print("\nNormalized Head Layer rank4")
    norm_head_rank40 = normalized_layer_rank(h_layer_rank4, alpha=1.1)
    # norm_head_rank41 = normalized_layer_rank(h_layer_rank4, alpha=1.2)
    # norm_head_rank42 = normalized_layer_rank(h_layer_rank4, alpha=1.3)
    print("\nLayer ratio4")
    ratio40 = rank_to_ratio(norm_neuron_rank40, 0.429) + rank_to_ratio(norm_head_rank40, 0.642)
    # ratio41 = rank_to_ratio(norm_neuron_rank41, 0.429) + rank_to_ratio(norm_head_rank41, 0.642)
    # ratio42 = rank_to_ratio(norm_neuron_rank42, 0.429) + rank_to_ratio(norm_head_rank42, 0.642)

    
    # 90% 的剪枝率
    print("\n90% Sparsity ratio00")
    args.neuron_sparsity = np.ones(12)-ratio00[:12]
    args.head_sparsity = np.ones(12)-ratio00[12:24]
    prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 50, device)
     

    # 80% 的剪枝率
    print("\n80% Sparsity ratio10")
    args.neuron_sparsity = np.ones(12)-ratio10[:12]
    args.head_sparsity = np.ones(12)-ratio10[12:24]
    prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 50, device)
    # print("\n80% Sparsity ratio11")
    # args.neuron_sparsity = np.ones(12)-ratio11[:12]
    # args.head_sparsity = np.ones(12)-ratio11[12:24]
    # prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 10, device)
    # print("\n80% Sparsity ratio12")
    # args.neuron_sparsity = np.ones(12)-ratio12[:12]
    # args.head_sparsity = np.ones(12)-ratio12[12:24]
    # prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 10, device)


    # 70% 的剪枝率
    print("\n70% Sparsity ratio20")
    args.neuron_sparsity = np.ones(12)-ratio20[:12]
    args.head_sparsity = np.ones(12)-ratio20[12:24]
    prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 50, device)
    # print("\n70% Sparsity ratio21")
    # args.neuron_sparsity = np.ones(12)-ratio21[:12]
    # args.head_sparsity = np.ones(12)-ratio21[12:24]
    # prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 10, device)   
    # print("\n70% Sparsity ratio22")
    # args.neuron_sparsity = np.ones(12)-ratio22[:12]
    # args.head_sparsity = np.ones(12)-ratio22[12:24]
    # prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 10, device) 


    # 60% 的剪枝率
    print("\n60% Sparsity ratio30")
    args.neuron_sparsity = np.ones(12)-ratio30[:12]
    args.head_sparsity = np.ones(12)-ratio30[12:24]
    prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 50, device)
    # print("\n60% Sparsity ratio31")
    # args.neuron_sparsity = np.ones(12)-ratio31[:12]
    # args.head_sparsity = np.ones(12)-ratio31[12:24]
    # prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 10, device)
    # print("\n60% Sparsity ratio32")
    # args.neuron_sparsity = np.ones(12)-ratio32[:12]
    # args.head_sparsity = np.ones(12)-ratio32[12:24]
    # prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 10, device) 
    

    # 50% 的剪枝率
    print("\n50% Sparsity ratio40")
    args.neuron_sparsity = np.ones(12)-ratio40[:12]
    args.head_sparsity = np.ones(12)-ratio40[12:24]
    prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 50, device)
    # print("\n50% Sparsity ratio41")
    # args.neuron_sparsity = np.ones(12)-ratio41[:12]
    # args.head_sparsity = np.ones(12)-ratio41[12:24]
    # prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 10, device)
    # print("\n50% Sparsity ratio42")
    # args.neuron_sparsity = np.ones(12)-ratio42[:12]
    # args.head_sparsity = np.ones(12)-ratio42[12:24]
    # prune(model, args, neuron_rank, head_rank, data_finetune, data_loader_val, 10, device)


    return 0


def save_data_loaders(data_loader_train, data_finetune, file_path_train='data_loader_train.pth', file_path_finetune='data_finetune.pth'):
    # 保存数据集对象
    torch.save(data_loader_train.dataset, file_path_train)  
    torch.save(data_finetune.dataset, file_path_finetune)  
    print("Data loaders saved to disk.")


def load_data_loaders(file_path_train='data_loader_train.pth', file_path_finetune='data_finetune.pth', batch_size=64, num_workers=4, worker_init_fn=None):
    # 加载数据集对象
    new_dataset = torch.load(file_path_train)
    new_dataset_finetune = torch.load(file_path_finetune)

    # 重新创建DataLoader
    new_data_loader = DataLoader(
        new_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers, 
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    new_data_finetune_loader = DataLoader(
        new_dataset_finetune, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers, 
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    print("Data loaders loaded from disk.")
    return new_data_loader, new_data_finetune_loader


def save_ranks(neuron_rank, head_rank, neuron_rank_path='neuron_rank.pth', head_rank_path='head_rank.pth'):
    # 保存神经元排序和头排序
    torch.save(neuron_rank, neuron_rank_path)
    torch.save(head_rank, head_rank_path)
    print(f"Neuron rank and head rank saved to {neuron_rank_path} and {head_rank_path}.")


def load_ranks(neuron_rank_path='neuron_rank.pth', head_rank_path='head_rank.pth'):
    # 加载神经元排序和头排序
    if os.path.exists(neuron_rank_path) and os.path.exists(head_rank_path):
        neuron_rank = torch.load(neuron_rank_path)
        head_rank = torch.load(head_rank_path)
        print("Neuron rank and head rank loaded from disk.")
        return neuron_rank, head_rank
    else:
        print("Rank files not found.")
        return None, None


def fine_tune(model, data_loader_train, data_loader_val, epochs, device):
    model = model.to(device)
    best_acc = 0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        # Training phase
        model.train()
        for inputs, labels in data_loader_train:
            inputs, labels = inputs.to(device), labels.to(device)           
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Evaluation phase
        top1, top5, _, _ = gp_evaluate(data_loader_val, model, device)
        if top1 > best_acc:
            best_acc = top1
        print(f"Epoch {epoch}: Top1 {top1}; Top5 {top5}")
        
    return best_acc


def prune(model, args, neuron_rank, head_rank, data_loader_train, data_loader_val, epoch, device): 
    tmp_model = copy.deepcopy(model)

    neuron_mask = mlp_neuron_mask(tmp_model, args.neuron_sparsity, neuron_rank)       
    mlp_neuron_prune(tmp_model, neuron_mask)
    print('Neuron sparsity', check_neuron_sparsity(tmp_model))

    head_mask = attn_head_mask(tmp_model, args.head_sparsity, head_rank)
    attn_head_prune(tmp_model, head_mask)
    print('Head sparsity', check_head_sparsity(tmp_model))

    print("Original")
    top1 = gp_evaluate(data_loader_val, tmp_model, device)
    print(top1)
    best_acc = fine_tune(tmp_model, data_loader_train, data_loader_val, epoch, device)
    print(f"Best Acc: {best_acc}")

    mlp_neuron_restore(tmp_model)
    attn_head_restore(tmp_model)

    # model = copy.deepcopy(opriginal_model)


@torch.no_grad()
def gp_evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_meter = AverageMeter()
    latency_meter = AverageMeter()

    # switch to evaluation mode
    model.eval()

    for _, (images, target) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 确保 CUDA 操作开始前同步
        torch.cuda.synchronize()
        start_time = time.time()

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # 确保所有 CUDA 操作完成
        torch.cuda.synchronize()
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # 转换为毫秒
        latency_meter.update(latency)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]

        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        loss_meter.update(loss.item(), batch_size)

    return top1.avg, top5.avg, loss_meter.avg, latency_meter.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)

