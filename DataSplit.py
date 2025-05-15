import os
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
    

def data_iid(dataset, num_users):
    num_clients = num_users
    num_data = len(dataset)
    data_per_client = num_data // num_clients

    indices = torch.randperm(len(dataset))
    shuffled_dataset = data.Subset(dataset, indices)

    client_datasets = []
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        client_data = data.Subset(shuffled_dataset, range(start, end))
        client_datasets.append(client_data)

    data_loader = []
    for i in range(num_clients):
        data_loader.append(data.DataLoader(
            client_datasets[i], 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))

    return data_loader


def data_iid_with_public(dataset, num_users):
    num_clients = num_users
    num_data = len(dataset)

    # Calculate the number of public data samples
    num_public_data = int(0.2 * num_data)
    num_private_data = num_data - num_public_data

    # Shuffle the dataset indices
    indices = torch.randperm(num_data)
    
    # Split the indices for public and private datasets
    public_indices = indices[:num_public_data]
    private_indices = indices[num_public_data:]

    # Create the public dataset
    public_dataset = data.Subset(dataset, public_indices)

    # Create a data loader for the public dataset
    public_data_loader = data.DataLoader(
        public_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    data_per_client = num_private_data // num_clients

    shuffled_private_dataset = data.Subset(dataset, private_indices)

    client_datasets = []
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        client_data = data.Subset(shuffled_private_dataset, range(start, end))
        client_datasets.append(client_data)

    data_loaders = []
    for i in range(num_clients):
        data_loaders.append(data.DataLoader(
            client_datasets[i], 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))

    return data_loaders, public_data_loader


def data_noniid(dataset, num_users):
    num_clients = num_users
    num_classes = len(dataset.classes)
    classes_per_client = num_classes // num_clients

    client_datasets = []
    for i in range(num_clients):
        start_class = i * classes_per_client
        end_class = (i + 1) * classes_per_client
        client_classes = dataset.classes[start_class:end_class]
        
        client_indices = [idx for idx, (_, label) in enumerate(dataset.samples) if label in client_classes]
        client_data = data.Subset(dataset, client_indices)
        client_datasets.append(client_data)

    data_loader = []
    for i in range(num_clients):
        data_loader.append(data.DataLoader(
            client_datasets[i], 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))

    return data_loader


def data_noniid_with_public(dataset, num_users):   
    num_clients = num_users
    num_data = len(dataset)
    num_classes = len(dataset.classes)
    classes_per_client = num_classes // num_clients

    # Calculate the number of public data samples
    num_public_data = int(0.2 * num_data)
    num_private_data = num_data - num_public_data

    # Shuffle the dataset indices
    indices = torch.randperm(num_data)

    # Split the indices for public and private datasets
    public_indices = indices[:num_public_data]
    private_indices = indices[num_public_data:]

    # Create the public dataset
    public_dataset = data.Subset(dataset, public_indices)

    # Create a data loader for the public dataset
    public_data_loader = data.DataLoader(
        public_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    client_datasets = []
    for i in range(num_clients):
        start_class = i * classes_per_client
        end_class = (i + 1) * classes_per_client
        client_classes = dataset.classes[start_class:end_class]
        
        client_indices = [idx for idx, (_, label) in enumerate(dataset.samples) 
                        if idx in private_indices and label in client_classes]
        client_data = data.Subset(dataset, client_indices)
        client_datasets.append(client_data)

    data_loader = []
    for i in range(num_clients):
        data_loader.append(data.DataLoader(
            client_datasets[i], 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))
    
    return data_loader, public_data_loader


def data_dirichlet(dataset, num_users):
    num_clients = num_users
    num_classes = len(dataset.classes)
    
    client_datasets = [[] for _ in range(num_clients)]
    
    # Get labels for all data
    all_labels = [label for _, label in dataset.samples]

    # Dirichlet distribution parameter
    alpha = 0.5

    # Generate Dirichlet distribution for each class
    for c in range(num_classes):
        class_indices = [idx for idx, label in enumerate(all_labels) if label == c]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_indices = np.split(class_indices, proportions)
        
        for i in range(num_clients):
            client_datasets[i].extend(client_indices[i])

    data_loader = []
    for i in range(num_clients):
        client_data = data.Subset(dataset, client_datasets[i])
        data_loader.append(data.DataLoader(
            client_data, 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))

    return data_loader


def data_dirichlet_with_public(dataset, num_users):
    num_clients = num_users
    num_data = len(dataset)
    num_classes = len(dataset.classes)
    
    # Calculate the number of public data samples
    num_public_data = int(0.2 * num_data)
    num_private_data = num_data - num_public_data

    # Shuffle the dataset indices
    indices = torch.randperm(num_data)

    # Split the indices for public and private datasets
    public_indices = indices[:num_public_data]
    private_indices = indices[num_public_data:]

    # Create the public dataset
    public_dataset = data.Subset(dataset, public_indices)

    # Create a data loader for the public dataset
    public_data_loader = data.DataLoader(
        public_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Get labels for private data
    private_labels = [dataset.samples[idx][1] for idx in private_indices]

    # Dirichlet distribution parameter
    alpha = 0.1
   
    # Generate Dirichlet distribution for each class
    client_datasets = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_indices = [idx for idx, label in zip(private_indices, private_labels) if label == c]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_indices = np.split(class_indices, proportions)
        
        for i in range(num_clients):
            client_datasets[i].extend(client_indices[i])

    data_loader = []
    for i in range(num_clients):
        client_data = data.Subset(dataset, client_datasets[i])
        data_loader.append(data.DataLoader(
            client_data, 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))
    
    return data_loader, public_data_loader


# 如下是本工作用到的函数
def load_imagenet_subset(root, class_indices, batch_size=64, num_workers=4, worker_init=None, train=True):
      
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载完整的 ImageNet 数据集
    subset_folder = 'train' if train else 'val'
    full_dataset = datasets.ImageFolder(root=os.path.join(root, subset_folder), transform=transform)

    # 过滤数据集以仅包含指定的类别
    subset_samples = [sample for sample in full_dataset.samples if sample[1] in class_indices]

    # 创建子集的数据集对象
    subset_dataset = datasets.DatasetFolder(
        root=root,
        loader=full_dataset.loader,
        extensions=full_dataset.extensions,
        transform=transform
    )

    # 将过滤后的样本集赋给子集数据集的 samples 属性
    subset_dataset.samples = subset_samples

    # 创建数据加载器
    subset_loader = data.DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        worker_init_fn=worker_init,
        pin_memory=True)

    return subset_loader


def load_imagenet_random(root, data_number, batch_size=64, num_workers=4, worker_init=None, train=True):
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载完整的 ImageNet 数据集
    folder = 'train' if train else 'val'
    dataset = datasets.ImageFolder(root=os.path.join(root, folder), transform=transform)
    
    # 随机采样 data_number 个样本索引
    total_samples = len(dataset)
    selected_indices = np.random.choice(total_samples, min(data_number, total_samples), replace=False)
    random_dataset = data.Subset(dataset, selected_indices)
    
    # 创建数据加载器
    random_loader = data.DataLoader(
        random_dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        worker_init_fn=worker_init,
        pin_memory=True)

    return random_loader


def load_imagenet(root, batch_size=64, num_workers=4, worker_init=None, train=True):

    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    folder = 'train' if train else 'val'
    dataset = datasets.ImageFolder(root=os.path.join(root, folder), transform=transform)

    if train:
        loader = data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=num_workers, 
            worker_init_fn=worker_init,
            pin_memory=True)
    else:
        loader = data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=num_workers, 
            worker_init_fn=worker_init,
            pin_memory=True)

    return loader


def load_cifar100_subset(root, class_indices, batch_size=64, num_workers=4, worker_init=None, train=True):
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    dataset = datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    subset_indices = [i for i in range(len(dataset)) if dataset.targets[i] in class_indices]
    subset_dataset = Subset(dataset, subset_indices)
    loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers, worker_init_fn=worker_init, pin_memory=True)
    return loader


def load_cifar100_random(root, data_number, batch_size=64, num_workers=4, worker_init=None, train=True):
    transform = transforms.Compose([
    transforms.Resize(224),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    dataset = datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    total_samples = len(dataset)
    selected_indices = np.random.choice(total_samples, min(data_number, total_samples), replace=False)
    random_dataset = Subset(dataset, selected_indices)
    loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers, worker_init_fn=worker_init, pin_memory=True)
    return loader


def load_cifar100(root, batch_size=64, num_workers=4, worker_init=None, train=True):
    transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    dataset = datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers, worker_init_fn=worker_init, pin_memory=True)
    return loader

