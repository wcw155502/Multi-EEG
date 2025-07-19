# -*- coding: utf-8 -*-       
# __@Time__    : 2025-07-15 21:53   
# __@Author__  : www             
# __@File__    : dataloader.py        
# __@Description__ :
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class DeapDataset(Dataset):
    """DEAP数据集加载器，用于加载预处理后的EEG、EOG、EMG数据和标签"""

    def __init__(self, eeg, eog, emg, v_labels, a_labels):
        """
        初始化Deap数据集
        :param eeg:
        :param eog:
        :param emg:
        :param v_labels:
        :param a_labels:
        """
        self.eeg = eeg
        self.eog = eog
        self.emg = emg
        self.v_labels = v_labels
        self.a_labels = a_labels

        print(f"数据集初始化完成，样本数: {self.eeg.shape[0]}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.eeg)

    def __getitem__(self, idx):
        """获取单个样本"""
        return {
            'eeg': self.eeg[idx],
            'eog': self.eog[idx],
            'emg': self.emg[idx],
            'valence': self.v_labels[idx],
            'arousal': self.a_labels[idx]
        }


def create_deap_datasets(data_dir, train_ratio=0.7, random_seed=42):
    """
    创建Deap数据集的训练集和测试集
    :param data_dir: 预处理数据所在目录
    :param train_ratio: 训练集占比
    :param random_seed: 随机种子
    :return: train_dataset: 训练集  test_dataset: 测试集
    """

    # 加载预处理数据
    data_file = os.path.join(data_dir, 'eeg_eog_emg_data.pt')
    labels_file = os.path.join(data_dir, 'valence_arousal_labels.pt')

    # 检查文件是否存在
    for file_path in [data_file, labels_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到数据文件: {file_path}")

    # 加载数据
    data = torch.load(data_file)
    labels = torch.load(labels_file)

    # 提取数据和标签
    eeg = data['eeg']
    eog = data['eog']
    emg = data['emg']
    v_labels = labels['valence']
    a_labels = labels['arousal']

    dataset = DeapDataset(eeg, eog, emg, v_labels, a_labels)

    # 数据划分
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    # 设置随机种子，确保划分可重复
    torch.manual_seed(random_seed)

    # 划分数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def create_deap_dataloaders(data_dir, train_batch_size=32, test_batch_size=32, train_ratio=0.7, random_seed=42,
                            num_workers=0):
    """
    创建DEAP数据集的训练和测试DataLoader
    :param data_dir:
    :param train_batch_size: 训练集批次大小
    :param test_batch_size: 测试集批次大小
    :param train_ratio:
    :param random_seed:
    :param num_workers: 数据加载的工作线程数
    :return: train_loader: 训练集DataLoader  test_loader: 测试集DataLoader
    """

    # 创建训练集和测试集
    train_dataset, test_dataset = create_deap_datasets(data_dir, train_ratio, random_seed)

    # 创建DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    # 测试DataLoader
    data_dir = f'D:\Dataset\DEAP\deap_preprocessed'  # 替换为你的数据目录
    batch_size = 16

    # 创建DataLoader
    train_loader, test_loader = create_deap_dataloaders(data_dir, train_batch_size=batch_size,
                                                        test_batch_size=batch_size)

    # 打印数据加载器信息
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")

    # 查看一个批次的数据
    batch = next(iter(train_loader))
    print("\n第一个训练批次的数据形状:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
