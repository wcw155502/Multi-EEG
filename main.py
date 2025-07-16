# -*- coding: utf-8 -*-       
# __@Time__    : 2025-07-16 11:45   
# __@Author__  : www             
# __@File__    : main.py        
# __@Description__ :
from data.dataloader import create_deap_dataloaders  # 导入之前创建的dataloader
import torch
import torch.nn as nn
from train.train import Trainer
from models.model2 import EDMERNet  # 导入模型
def main():
    # 配置
    config = {
        'lr': 0.001,
        'batch_size': 256,
        'epochs': 200,
        'model': 'EDMERNet',
        'dataset': 'DEAP'
    }

    # 初始化WandB（取消注释以启用）
    # wandb.init(
    #     project='MultiTransformerEmotion',
    #     config=config
    # )

    #数据根目录
    data_dir = f'D:\Dataset\DEAP\deap_preprocessed'

    # 创建DataLoader
    train_dataloader, test_dataloader = create_deap_dataloaders(
        data_dir=data_dir,
        train_batch_size=config['batch_size'],
        test_batch_size=config['batch_size'],
        train_ratio=0.7,
        random_seed=42
    )

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型、损失函数和优化器
    model = EDMERNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        is_wandb=False  # 设置为True以启用WandB记录
    )

    # 训练模型
    trainer.train(num_epochs=config['epochs'])

    # 完成WandB记录（取消注释以启用）
    # wandb.finish()


if __name__ == "__main__":
    main()