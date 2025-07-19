# -*- coding: utf-8 -*-
# __@Time__    : 2025-07-14 17:11
# __@Author__  : www
# __@File__    : train.py
# __@Description__ : 多模态情感识别训练脚本

import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score
import wandb


class Trainer:
    """模型训练器，负责模型的训练和验证"""

    def __init__(self, model, train_dataloader, test_dataloader, criterion, optimizer, device, is_wandb=False, target_label="valence"):
        """
        初始化训练器
        :param model: 待训练的模型
        :param train_dataloader: 训练数据加载器
        :param test_dataloader: 测试数据加载器
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param device: 计算设备
        :param is_wandb: 是否使用WandB记录训练过程
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.is_wandb = is_wandb
        self.target_label = target_label

    def train(self, num_epochs):
        """
        训练模型
        :param num_epochs: 训练轮数
        :return:
        """
        # 最高准确率定义
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            # 训练阶段
            train_loss, train_metrics = self._train_epoch()

            # 验证阶段
            test_loss, test_metrics = self.evaluate(self.test_dataloader)

            # 打印训练信息
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train: Loss={train_loss:.4f}, Accuracy={train_metrics['accuracy']:.4f}, "
                  f"F1={train_metrics['f1']:.4f}, Recall={train_metrics['recall']:.4f}")
            print(f"Test:  Loss={test_loss:.4f}, Accuracy={test_metrics['accuracy']:.4f}, "
                  f"F1={test_metrics['f1']:.4f}, Recall={test_metrics['recall']:.4f}")
            print("-" * 50)

            # 记录到WandB
            if self.is_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'train_recall': train_metrics['recall'],
                    'test_loss': test_loss,
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1'],
                    'test_recall': test_metrics['recall']
                })

            # 保存最佳模型
            if test_metrics['accuracy'] > best_accuracy:
                best_accuracy = test_metrics['accuracy']
                # self._save_model(f'model_best.pth')

        print(f"训练完成！最佳测试准确率: {best_accuracy:.4f}")

    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        for batch in self.train_dataloader:
            # 数据移至设备
            eeg, eog, emg = batch["eeg"].to(self.device), batch["eog"].to(self.device), batch["emg"].to(self.device)
            labels = batch[self.target_label].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(eeg, eog, emg)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录损失和预测结果
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # 计算平均损失和评估指标
        avg_loss = total_loss / len(self.train_dataloader)
        metrics = self._calculate_metrics(all_labels, all_predictions)

        return avg_loss, metrics

    def evaluate(self, dataloader):
        """
        评估模型性能
        :param dataloader: 评估数据加载器
        :return: 平均损失和包含准确率、F1分数和Recall的字典
        """
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                # 数据移至设备
                eeg, eog, emg = batch["eeg"].to(self.device), batch["eog"].to(self.device), batch["emg"].to(self.device)
                labels = batch[self.target_label].to(self.device)

                # 前向传播
                outputs = self.model(eeg, eog, emg)
                loss = self.criterion(outputs, labels)

                # 记录损失和预测结果
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # 计算平均损失和评估指标
        avg_loss = total_loss / len(dataloader)
        metrics = self._calculate_metrics(all_labels, all_predictions)

        return avg_loss, metrics

    def _calculate_metrics(self, labels, predictions):
        """
        计算评估指标
        :param labels: 真实标签
        :param predictions: 预测标签
        :return: 包含准确率、F1分数和Recall的字典
        """
        accuracy = np.mean(np.array(labels) == np.array(predictions))
        f1 = f1_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall
        }

    def _save_model(self, path):
        """
        保存模型"
        :param path: 保存地址
        :return:
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"模型已保存至 {path}")
