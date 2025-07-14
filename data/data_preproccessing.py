import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import numpy as np


class DEAP(Dataset):
    """DEAP数据集加载器"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.sessions = os.listdir(dataset_path)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int):
        file_path = os.path.join(self.dataset_path, self.sessions[index])
        session = sio.loadmat(file_path)
        data, labels = session['data'], session['labels']

        labels = (labels >= 5.0).astype(np.int64)

        # 划分基线数据和试验数据
        baseline = data[:, :36, :128 * 3]
        data = data[:, :36, 128 * 3:]

        return baseline, data, labels


class Preprocessor:
    """数据预处理类"""

    def __init__(self, file_path):
        super(Preprocessor, self).__init__()
        self.dataset = DEAP(file_path)

    def preprocess_data(self, is_map: bool = False):
        """
        数据预处理
        :param is_map: eeg数据是否映射为矩阵
        :return: eeg, eog, emg, v_labels, a_labels
        """
        dataset = self.dataset

        array_data = []
        array_labels = []

        for i in range(dataset.__len__()):
            base, data, labels = dataset.__getitem__(i)

            # 基线处理
            base_mean = self.process_baseline(base)

            # 基线校准 + 正则化
            data = self.process_trial(data, base_mean)

            # 通道映射
            # if is_map:
            #     eeg_mapped = processor.map_eeg_channels(data[:, :32, :])

            # 数据分割
            data = self.data_segment(data, 1, 128)

            # 标签处理
            labels = self.process_labels(labels)

            # 数据添加集合
            array_data.append(data)
            array_labels.append(labels)

        data = np.concatenate(array_data)
        labels = np.concatenate(array_labels)

        eeg = torch.FloatTensor(data[:, 0:32, :])
        eog = torch.FloatTensor(data[:, 32:34, :])
        emg = torch.FloatTensor(data[:, 34:36, :])

        v_labels, a_labels = labels[:, 0], labels[:, 1]

        return eeg, eog, emg, v_labels, a_labels

    @staticmethod
    def process_baseline(baseline: np.ndarray) -> np.ndarray:
        """
        处理基线信号（3段平均）
        :param baseline:基线数据(40, 32, 384) or (40, 4, 384) 384 = 128 * 3
        :return:(40, 32, 128)的平均值
        """
        assert baseline.shape[2] == 384, "基线数据长度应为384"
        segments = np.array_split(baseline, 3, axis=2)
        return np.mean(segments, axis=0).astype(np.float32)

    @staticmethod
    def process_trial(trial: np.ndarray, baseline_mean: np.ndarray) -> np.ndarray:
        """
        处理试验信号（基线校正+标准化）
        :param trial: 实验数据
        :param baseline_mean: 基线数据
        :return:
        """
        assert trial.shape[2] % 128 == 0, "试验数据长度应为128的整数倍"
        segments = np.array_split(trial, 60, axis=2)

        # 基线校正
        corrected = [seg - baseline_mean for seg in segments]
        processed = np.concatenate(corrected, axis=2)

        # 标准化
        mean = np.mean(processed)
        std = np.std(processed) + 1e-8
        return ((processed - mean) / std).astype(np.float32)

    @staticmethod
    def data_segment(trial: np.ndarray, segment_duration: int = 1, sampling_rate: int = 128) -> np.ndarray:
        """
        按时间对数据进行分割
        :param trial: 实验数据
        :param segment_duration: 分段时长（默认1秒）
        :param sampling_rate: 采样率（默认128Hz）
        :return: 处理后的信号，形状为(trials*segments, channels, segment_duration * sampling_rate)
        """
        # 计算分段参数
        segment_size = segment_duration * sampling_rate
        total_segments = trial.shape[-1] // segment_size

        # 分割
        segments = np.array_split(trial, total_segments, axis=-1)

        # 维度变换：从(trials, channels, segments*size) → (trials * segments, channels, size
        return np.concatenate(segments, axis=0).astype(np.float32)

    @staticmethod
    def map_eeg_channels(eeg_data: np.ndarray) -> np.ndarray:
        """
        将eeg数据映射到二维矩阵
        :param eeg_data:
        :return:
        """
        electrode_map = np.array([
            [0, 3], [1, 3], [2, 2], [2, 0],
            [3, 1], [3, 3], [4, 2], [4, 0],
            [5, 1], [5, 3], [6, 2], [6, 0],
            [7, 3], [8, 3], [8, 4], [6, 4],
            [0, 5], [1, 5], [2, 4], [2, 6],
            [2, 8], [3, 7], [3, 5], [4, 4],
            [4, 6], [4, 8], [5, 7], [5, 5],
            [6, 6], [6, 8], [7, 5], [8, 5]
        ])

        mapped = np.zeros((eeg_data.shape[0], 9, 9, eeg_data.shape[2]))
        for trial in range(eeg_data.shape[0]):
            for ch in range(32):
                x, y = electrode_map[ch]
                mapped[trial, x, y] = eeg_data[trial, ch]
        return mapped

    @staticmethod
    def process_labels(labels):
        """
        标签处理， 只留valence和arousal标签
        :param labels:
        :return:
        """
        labels = labels[:, :2]
        labels = np.tile(labels, (60, 1))
        # 第一个是valence， 第二个是arousal
        return labels


if __name__ == '__main__':

    filepath = f'D:\Code\Muilti-EEG\data\data_preprocessed_matlab'
    dataprocessor = Preprocessor(filepath)
