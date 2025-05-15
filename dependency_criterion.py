import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


num_blocks = 12

# nn.Linear indices
attn_qkv = [4*i+1 for i in range(num_blocks)]
attn_proj = [4*i+2 for i in range(num_blocks)]
mlp_fc1 = [4*i+3 for i in range(num_blocks)]
mlp_fc2 = [4*i+4 for i in range(num_blocks)]


def mlp_neuron_rank(model, train_loader, device):
    score = {}

    with torch.no_grad():
        relevance = HSICLoss(y_kernel='linear', mean_sub=True).cuda()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 50:
                break
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            idx = 0
            for m in model.modules():
                if 'Mlp' in str(m) and 'Attention' not in str(m):
                    X_ = m.neuron_output # batch x seq x embed

                    # 相关性分数 max
                    hsic = []
                    for H1 in range(X_.shape[-1]):
                        hsic.append(relevance(X_[:,:,H1], F.softmax(output, dim=-1)).item())                   
                    hsic = np.array(hsic)
                    hsic = (hsic - np.min(hsic)) / (np.max(hsic) - np.min(hsic))
                    # 激活值分数 max
                    act = np.sum(X_.abs().detach().cpu().numpy(), axis=(0,1))
                    act = (act - np.min(act)) / (np.max(act) - np.min(act))
                    # 综合分数
                    temp = (0.2*hsic + 0.8*act).tolist()

                    if batch_idx == 0:
                        score[str(idx)] = np.array(temp)
                    else:
                        score[str(idx)] += np.array(temp)
                    idx += 1
                    continue
                
    # rank是一个列表, 其中每个元素是模型中每一层MLP排好序的神经元索引列表
    rank = [np.argsort(score[str(idx)]) for idx in range(len(score))]

    return rank


def mlp_neuron_mask(model, ratio, rank):
    idx = 0
    neuron_mask = []
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            num_keep = int(m.hidden_features * (1 - ratio[idx]))
            # [::-1]是返回逆序列表的意思
            arg_max_rev = rank[idx][::-1][:num_keep]
            mask = torch.zeros(m.hidden_features)
            mask[arg_max_rev.tolist()] = 1
            neuron_mask.append(mask)
            idx += 1
            continue
    return neuron_mask


def mlp_neuron_prune(model, neuron_mask):
    idx = 0
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            m.gate = neuron_mask[idx]
            idx += 1
            continue


def mlp_neuron_restore(model):
    idx = 0
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            temp = m.gate.detach().clone()
            m.gate = torch.ones(temp.shape[0])
            idx += 1
            continue


def check_neuron_sparsity(model):
    ratio = []
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            ratio.append(torch.sum(m.gate == 0).item() / m.gate.shape[0])
            continue
    return ratio


def attn_head_rank(model, train_loader, device):   
    score = {}

    with torch.no_grad():
        relevance = HSICLoss(y_kernel='linear', mean_sub=True).cuda()
        redundancy = HSICLoss(y_kernel='rbf', mean_sub=False).cuda()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 50:
                break
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            idx = 0
            for m in model.modules():
                if 'Attention' in str(m) and 'Mlp' not in str(m):
                    X_ = m.head_output # batch x seq x head*embed
                    X_head = X_.reshape(X_.shape[0], X_.shape[1], m.num_heads, m.head_dim)
                    head_list = []
                    for idx_head in range(m.num_heads):
                        head_list.append(torch.mean(X_head[:,:,idx_head,:], dim=-1))
                    
                    relevance_list = []
                    redundancy_list = []
                    for H1 in range(X_.shape[2]):
                        # 相关性分数 max
                        relevance_count = relevance(X_[:,:,H1], F.softmax(output, dim=-1)).item()
                        relevance_list.append(relevance_count)
                        # 冗余度分数 min
                        redundancy_count = 0
                        for H2 in range(m.num_heads):
                            if H1 not in np.arange(H2*m.head_dim, (H2+1)*m.head_dim):
                                redundancy_count += redundancy(X_[:,:,H1], head_list[H2]).item()
                        redundancy_count /= (len(head_list)-1)
                        redundancy_list.append(redundancy_count)
                    # 归一化处理
                    relevance_list = np.array(relevance_list)
                    relevance_list = (relevance_list - np.min(relevance_list)) / (np.max(relevance_list) - np.min(relevance_list))
                    redundancy_list = np.array(redundancy_list)
                    redundancy_list = (redundancy_list - np.min(redundancy_list)) / (np.max(redundancy_list) - np.min(redundancy_list))
                    redundancy_list = -redundancy_list + 1
                    # 综合分数
                    temp = (0.8*relevance_list + 0.2*redundancy_list).tolist()

                    if batch_idx == 0:
                        score[str(idx)] = np.array(temp)
                    else:
                        score[str(idx)] += np.array(temp)
                    idx += 1
                    continue

    # rank是一个列表, 其中每个元素是模型中每一层ATTN排好序的神经元索引列表
    rank = [(np.argsort(score[str(idx)])) for idx in range(len(score))]

    return rank


def attn_head_mask(model, ratio, rank):
    idx = 0
    head_mask = []
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            num_keep = int(m.num_heads * m.head_dim * (1 - ratio[idx]))
            # [::-1]是返回逆序列表的意思
            arg_max_rev = rank[idx][::-1][:num_keep]
            mask = torch.zeros(m.num_heads * m.head_dim)
            mask[arg_max_rev.tolist()] = 1
            head_mask.append(mask)
            idx += 1
            continue
    return head_mask


def attn_head_prune(model, head_mask):
    idx = 0
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            m.gate = head_mask[idx]
            idx += 1
            continue


def attn_head_restore(model):
    idx = 0
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            temp = m.gate.detach().clone()
            m.gate = torch.ones(temp.shape[0])
            idx += 1
            continue


def check_head_sparsity(model):
    ratio = []
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            ratio.append(torch.sum(m.gate == 0).item() / m.gate.shape[0])
            continue
    return ratio


def layer_rank(model, train_loader, device, sparsity, neuron_rank, head_rank):
    model_select = copy.deepcopy(model)
    neuron_layer_rank = np.zeros(12)
    head_layer_rank = np.zeros(12)

    for i in range(12):
        print("Neuron Layer", i)
        neuron_sparsity = np.zeros(12)
        neuron_sparsity[i] = sparsity

        neuron_mask = mlp_neuron_mask(model_select, neuron_sparsity, neuron_rank) 
        mlp_neuron_prune(model_select, neuron_mask)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 50:
                    break
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                output_select = model_select(data)
                score_select = kl_divergence(F.softmax(output, dim=-1), F.softmax(output_select, dim=-1)).item()
                neuron_layer_rank[i] += score_select
        mlp_neuron_restore(model_select)

    for i in range(12):
        print("Head Layer", i)
        head_sparsity = np.zeros(12)
        head_sparsity[i] = sparsity

        head_mask = attn_head_mask(model_select, head_sparsity, head_rank)
        attn_head_prune(model_select, head_mask)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 50:
                    break
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                output_select = model_select(data)
                score_select = kl_divergence(F.softmax(output, dim=-1), F.softmax(output_select, dim=-1)).item()
                head_layer_rank[i] += score_select
        attn_head_restore(model_select)

    print("\nNeuron Layer Rank:\n", neuron_layer_rank)
    print("\nHead Layer Rank:\n", head_layer_rank)
    rank = np.hstack((neuron_layer_rank, head_layer_rank))
    print("\nLayer Rank:\n", rank)

    return rank, neuron_layer_rank, head_layer_rank


def normalized_layer_rank(rank, alpha = 1):
    # alpha是一个超参数，用于控制softmax的温度
    exp_values = np.exp(rank * alpha)
    norm_rank = exp_values / np.sum(exp_values)
    print("\nNormalized Layer Rank:\n", norm_rank) 

    return norm_rank


def rank_to_ratio(rank, num_keep):    
    ratio = []     
    obj = num_keep * 12
    base = sum(rank)
    for i in range(12):
        ratio.append(rank[i] * obj / base) 
        if ratio[i] > 1:
            ratio[i] = 1
        if ratio[i] < 0:
            ratio[i] = 0
    print("\nPruning Ratio:\n", ratio)        
    return ratio               


def token_layer_rank(model, train_loader, device):
    model_select = copy.deepcopy(model)
    output = None
    layer_score = []
    layer = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 50:
                break
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)            
            idx = 0
            for it in np.ndindex(3, 3, 3):
                tmp = np.add(list(it), [4, 7, 10])
                tmp = list(tmp)
                set_token_selection_layer(model_select, 0.3, tmp)
                
                output_select = model_select(data)
                score_select = kl_divergence(F.softmax(output, dim=-1), F.softmax(output_select, dim=-1)).item()
                
                if batch_idx == 0:
                    layer_score.append([tmp, score_select])
                else:
                    layer_score[idx][1] += score_select
                
                reset_token_selection_layer(model_select, tmp)
                idx += 1

    layer_score = sorted(layer_score, key=lambda x: x[1])
    layer = layer_score[0][0]

    return layer


def set_token_selection_layer(model, token_sparsity=[0.5,0.5,0.5], layer=[4,7,10]):
    idx = 1
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            if idx in layer:
                m.token_prune_ratio = token_sparsity[layer.index(idx)]
            idx += 1
            continue


def reset_token_selection_layer(model, layer=[4,7,10]):
    idx = 1
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            if idx in layer:
                m.token_prune_ratio = 0
            idx += 1
            continue	
    for m in model.modules():
        if 'Block' in str(m) and 'ModuleList' not in str(m):
            m.ema_cls_attn = None


# 相关性分析
def kl_divergence(p, q):
    """
    计算两个概率分布p和q之间的KL散度
    :param p: 第一个概率分布 (Tensor)
    :param q: 第二个概率分布 (Tensor)
    :return: KL散度 (Tensor)
    """
    p = p / p.sum()
    q = q / q.sum()
    return F.kl_div(p.log(), q, reduction='batchmean')


def center(X):
    mean_col = torch.mean(X, dim=0, keepdim=True)
    mean_row = torch.mean(X, dim=1, keepdim=True)
    mean_all = torch.mean(X)
    return X - mean_col - mean_row + mean_all
    

class GaussianKernel(nn.Module):
    def __init__(self, sigma):
        super(GaussianKernel, self).__init__()
        assert sigma > 0
        self.sigma = sigma

    def forward(self, x):
        X_inner = torch.matmul(x, x.t())
        X_norm = torch.diag(X_inner, diagonal=0)
        X_dist_sq = X_norm + torch.reshape(X_norm, [-1,1]) - 2 * X_inner
        return torch.exp( - X_dist_sq / (2 * self.sigma**2))


class LinearKernel(nn.Module):
    def __init__(self,):
        super(LinearKernel, self).__init__()

    def forward(self, x):
        return torch.matmul(x, x.t())
    

class HSICLoss(nn.Module):
    def __init__(self, y_kernel='linear', mean_sub=False):
        super(HSICLoss, self).__init__()

        self.kernelX_1 = GaussianKernel(1)
        self.kernelX_2 = GaussianKernel(2)
        self.kernelX_4 = GaussianKernel(4)
        self.kernelX_8 = GaussianKernel(8)
        self.kernelX_16 = GaussianKernel(16)

        self.y_kernel = y_kernel
        if self.y_kernel == 'linear':
            self.kernelY = LinearKernel()
        elif self.y_kernel == 'rbf':
            self.kernelY = None

        self.mean_sub = mean_sub

    def forward(self, x, y):
        # x: feature  y: softmax prediction
        if self.mean_sub is True:
            x = x - torch.mean(x, dim=0) / (torch.std(x, dim=0) + 1e-12)
            y = y - torch.mean(y, dim=0)

        G_X = center((self.kernelX_1(x) + self.kernelX_2(x) + self.kernelX_4(x) + self.kernelX_8(x) + self.kernelX_16(x))/5)

        if self.y_kernel == 'linear':
            G_Y = center(self.kernelY(y))
        elif self.y_kernel == 'rbf':
            G_Y = center((self.kernelX_1(y) + self.kernelX_2(y) + self.kernelX_4(y) + self.kernelX_8(y) + self.kernelX_16(y))/5)

        return torch.trace(torch.matmul(G_X, G_Y))
    
