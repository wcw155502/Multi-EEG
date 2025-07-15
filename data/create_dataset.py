# -*- coding: utf-8 -*-       
# __@Time__    : 2025-07-15 19:30   
# __@Author__  : www             
# __@File__    : create_dataset.py        
# __@Description__ :
import os
import torch
from data_preproccessing import Preprocessor


def save_processed_data(dataset_path, save_dir, is_map=False, overwrite=False):
    """
    预处理DEAP数据集并保存为PyTorch张量文件
    :param dataset_path: 原始数据集路径
    :param save_dir: 处理后数据保存路径
    :param is_map: 是否将EEG数据映射到二维矩阵
    :param overwrite: 是否覆盖已存在的文件
    :return:
    """

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 定义保存文件路径
    data_file = os.path.join(save_dir, 'eeg_eog_emg_data.pt')
    labels_file = os.path.join(save_dir, 'valence_arousal_labels.pt')

    # 检查文件是否已存在
    if os.path.exists(data_file) and os.path.exists(labels_file) and not overwrite:
        print("处理后的数据已存在，跳过预处理步骤。")
        return data_file, labels_file

    # 初始化预处理器
    processor = Preprocessor(dataset_path)

    # 执行预处理（显示进度条）
    print("开始预处理DEAP数据集...")
    eeg, eog, emg, v_labels, a_labels = processor.preprocess_data(is_map=is_map)

    # 合并数据和标签
    print("合并数据和标签...")
    data = {
        'eeg': eeg,
        'eog': eog,
        'emg': emg
    }

    labels = {
        'valence': v_labels,
        'arousal': a_labels
    }

    # 保存处理后的数据
    print("保存预处理数据到文件...")
    torch.save(data, data_file)
    torch.save(labels, labels_file)

    print(f"预处理完成！数据已保存到: {save_dir}")
    return data_file, labels_file


if __name__ == "__main__":
    # 配置路径
    dataset_path = f'D:\Code\Muilti-EEG\data\data_preprocessed_matlab'  # 替换为你的DEAP数据集路径
    save_dir = f'D:\Dataset\DEAP\deap_preprocessed'  # 处理后数据保存路径

    # 预处理并保存数据
    data_file, labels_file = save_processed_data(dataset_path, save_dir, is_map=False, overwrite=True)
