import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
from torchvision.transforms import transforms
import numpy as np
import cv2
import random
from sklearn.mixture import GaussianMixture
from collections import Counter
from PIL import Image, ImageDraw, ImageFilter
from collections import OrderedDict


class Head(nn.Module):
    def __init__(self, input_dim=128, output_dim=200):

        super(Head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):

        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, output_dim)
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class SimSiamFeatureExtractor(nn.Module):
    def __init__(self, backbone, projection_head, head):
       
        super(SimSiamFeatureExtractor, self).__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.head = head

    def forward(self, x):
        # 获取 backbone 的特征
        features = self.backbone(x)
        # 展平特征
        features = features.view(features.size(0), -1)
        # 通过投影头映射到 128 维
        projected_features = self.projection_head(features) 
        # 通过分类头映射到 200 维
        projected_features = self.head(projected_features)     
        return projected_features
    
    def get_projected_features(self, x):
        # 获取 backbone 的特征
        features = self.backbone(x)
        # 展平特征
        features = features.view(features.size(0), -1)
        # 通过投影头映射到 128 维
        projected_features = self.projection_head(features)
        return projected_features


class EdgeDevice:
    def __init__(self, dataset, batch_size=64, num_workers=4, worker_init=None):
        self.dataset = dataset                  # 设备数据集
        self.F = None                           # 最佳混合高斯分布
        self.selected_dataset = None            # 云侧为该设备挑选出来的数据集, 用于剪枝 
        self.selected_dataset_finetune = None   # 云侧为该设备挑选出来的数据集, 用于微调
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.worker_init = worker_init
        
    def Calculate_data_distribution(self):
        # 计算特征描述符
        print("Calculating deep learning descriptors\n")
        descriptors = compute_deep_learning_descriptors_from_dataloader(self.dataset)
        # 选择最佳的混合高斯分布
        print("Selecting best GMM\n")
        self.F, bic_scores = select_best_gmm(descriptors)
        # 输出最佳混合高斯分布参数
        # print("Edge device has data distribution:\n")
        # print("Weights:\n", self.F.weights_)
        # print("Means:\n", self.F.means_)
        # print("Covariances:\n", self.F.covariances_) 

    def set_selected_dataset(self, public_dataset):
        gmm = self.F
        new_dataset, new_dataset_finetune = select_images_gmm(public_dataset, gmm, self.batch_size*10)
        print("A new dataset has been created containing the number of images:", len(new_dataset))
        
        # 将新数据集转换为DataLoader
        new_data_loader = DataLoader(
                            new_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=True, 
                            drop_last=True, 
                            num_workers=self.num_workers, 
                            worker_init_fn=self.worker_init,
                            pin_memory=True)   
        new_data_finetune_loader = DataLoader(
                            new_dataset_finetune, 
                            batch_size=self.batch_size, 
                            shuffle=True, 
                            drop_last=True, 
                            num_workers=self.num_workers, 
                            worker_init_fn=self.worker_init,
                            pin_memory=True)
        
        self.selected_dataset = new_data_loader
        self.selected_dataset_finetune = new_data_finetune_loader
  

# 计算深度学习特征描述符
def compute_deep_learning_descriptors_from_dataloader(dataloader):
    descriptors_list = []
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的 ResNet18 并移除分类头
    backbone = models.resnet18()
    backbone.fc = nn.Identity()  # 去掉最后的分类层

    # 初始化投影头
    head = Head(input_dim=128, output_dim=200)
    projection_head = ProjectionHead(input_dim=512, hidden_dim=256, output_dim=128)

    # 创建 SimSiam 特征提取器模型
    state_dict = torch.load('../logs/extractor18.pth')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    simsiam_extractor = SimSiamFeatureExtractor(backbone, projection_head, head)
    simsiam_extractor.load_state_dict(new_state_dict)
    simsiam_extractor.eval()

    for images, _ in dataloader:        
        simsiam_extractor.to(device)
        images = images.to(device)

        # 获取特征
        descriptors = simsiam_extractor.get_projected_features(images)      
        # descriptors = F.softmax(descriptors, dim=1)
        descriptors_list.append(descriptors.detach().cpu().numpy())
        
    if descriptors_list:
        all_descriptors = np.vstack(descriptors_list)
        print("all_descriptors:", all_descriptors.shape)
        return all_descriptors
    else:
        return np.array([])


# 使用BIC选择最佳高斯分布数量的函数 
def select_best_gmm(descriptors, max_components=10):
    best_gmm = None
    lowest_bic = np.inf
    bic_scores = []
    num = 0

    for n in range(2, max_components + 1, 1):
        print("GMM with", n, "components\n")
        gmm = GaussianMixture(n_components=n, random_state=36)
        gmm.fit(descriptors)
        bic = gmm.bic(descriptors)
        bic_scores.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
            num = n
    print("Best GMM has", num, "components\n")
    return best_gmm, bic_scores


def select_images_gmm(images, gmm, num_samples=64):
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的 ResNet18 并移除分类头
    backbone = models.resnet18()
    backbone.fc = nn.Identity()  # 去掉最后的分类层

    # 初始化投影头
    head = Head(input_dim=128, output_dim=200)
    projection_head = ProjectionHead(input_dim=512, hidden_dim=256, output_dim=128)

    # 创建 SimSiam 特征提取器模型
    state_dict = torch.load('../logs/extractor18.pth')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    simsiam_extractor = SimSiamFeatureExtractor(backbone, projection_head, head)
    simsiam_extractor.load_state_dict(new_state_dict)
    simsiam_extractor.eval()

    select = []
    selected_images = []  
    selected_images_finetune = []
    print("Selecting images\n")

    for images, labels in images:           
        simsiam_extractor.to(device)
        images = images.to(device)

        # 获取特征
        descriptors = simsiam_extractor.get_projected_features(images) 
        # descriptors = F.softmax(descriptors, dim=1)         
        descriptors = descriptors.detach().cpu().numpy()

        for desc, image, label in zip(descriptors, images, labels):
            image_log_prob = gmm.score_samples(desc.reshape(1, -1))
            select.append((image_log_prob, (image.cpu(), label)))

    log_probs = [item[0] for item in select]
    # 计算最大值和最小值
    max_log_prob = max(log_probs)
    min_log_prob = min(log_probs)
    # 将 image_log_prob 映射到 0~1 之间
    def normalize_log_prob(log_prob, min_val, max_val):       
        if max_val == min_val:
            return 0.0 
        return (log_prob - min_val) / (max_val - min_val)
    normalized_select = [(normalize_log_prob(image_log_prob, min_log_prob, max_log_prob), (image, label))
                        for image_log_prob, (image, label) in select]
    

    # # 根据概率值决定选择的图像
    # idx_list = []
    # while True:        
    #     print(len(selected_images))              
    #     for i in range(len(select)):
    #         prob = np.exp(select[i][0])
    #         rand = np.random.rand()
    #         if rand < prob and i not in idx_list:
    #             if len(selected_images) < num_samples:                   
    #                 selected_images.append(select[i][1])
    #             selected_images_finetune.append(select[i][1])
    #             idx_list.append(i)

    #     if len(selected_images_finetune) >= num_samples*10:
    #         break

    # final_selected_images = random.sample(selected_images, int(num_samples))
    # final_selected_images_finetune = random.sample(selected_images_finetune, int(num_samples*10))


    # 选择概率最高的图像
    normalized_select.sort(key=lambda x: x[0], reverse=True)
    for i in range(num_samples):
        selected_images.append(normalized_select[i][1]) 
        # data=open("./select_descriptors.txt",'a')
        # data.write(f"Lable: {normalized_select[i][1][1]}\n")
        # data.close()   
    for i in range(num_samples*10):
        selected_images_finetune.append(normalized_select[i][1])
    
    final_selected_images = random.sample(selected_images, int(num_samples))
    final_selected_images_finetune = random.sample(selected_images_finetune, int(num_samples*10))

    return final_selected_images, final_selected_images_finetune

