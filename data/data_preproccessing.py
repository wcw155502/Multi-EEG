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
    """数据预处理"""

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
    def process_data(trial: np.ndarray, segment_duration: int = 1, sampling_rate: int = 128) -> np.ndarray:
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

        #分割
        segments = np.array_split(trial, total_segments, axis=-1)

        # 维度变换：从(trials, channels, segments*size) → (trials*segments, channels, size
        return np.concatenate(segments, axis=0).astype(np.float32)

    @staticmethod
    def map_eeg_channels(eeg_data: np.ndarray) -> np.ndarray:
        """将EEG通道映射到二维矩阵"""
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

class DataProcessor:
    """完成数据处理流程"""

    @staticmethod
    def process_data(file_path: str, is_map: bool = False, return_eeg: bool = True, return_other: bool = True):

        dataset = DEAP(file_path)
        processor = Preprocessor()

        eeg_data = []
        other_data = []
        v_label = []
        a_label = []

        for i in range(dataset.__len__()):
            base, data = dataset[i]

            #基线处理
            base_mean = processor.process_baseline(base)

            #基线校准 + 正则化
            data = processor.process_trial(data)

            #通道映射
            if is_map:
                eeg_mapped = processor.map_eeg_channels(data[:, :32, :])

            #数据分割


        return

    @staticmethod
    def process_labels(labels: np.ndarray) -> tuple:
        """处理效价和唤醒度标签"""
        labels = labels[:, :2]
        return labels[:, 0], labels[:, 1]



def baseline_sigal_processing(baseline_data):
    """
    将基线信号分割为三段，并计算每段的平均值
    :param baseline_data: 基线数据(40, 32, 384) or (40, 4, 384) 384 = 128 * 3
    :return: baseline_mean: (40, 32, 128)的平均值
    """
    assert baseline_data.shape[2] == 384

    segments = np.array_split(baseline_data, 3, axis=2)
    # baseline_mean = (baseline_segments[0] + baseline_segments[1] + baseline_segments[2]) / 3

    return np.mean(segments, axis=0).astype(np.float32)


def trail_signal_proccessing(trail_data, baseline_mean):
    """
    将没有基线信号的数据切割成60段，然后减去基线信号的平均值，得到预处理信号
    :param trail_data: 实验数据
    :return:
    """
    assert trail_data.shape == (40, 32, 7680) or (40, 4, 7680)
    assert baseline_mean.shape == (40, 32, 128) or (40, 4, 7680)
    trail_segments = np.array_split(trail_data, 60, axis=2)

    array_data = []

    for i, trail_segment in enumerate(trail_segments):  # 对每一s做 偏差 处理
        assert trail_segment.shape == (40, 32, 128) or (40, 4, 7680)
        per_trail_proccessing_data = trail_segment - baseline_mean
        array_data.append(per_trail_proccessing_data)

    trail_processing_data = np.concatenate(array_data, axis=2)

    assert trail_processing_data.shape == (40, 32, 7680) or (40, 4, 7680)
    mean = np.mean(trail_processing_data, axis=(0, 1, 2), keepdims=True)  # 计算每个特征的均值，保持维度 (1, 1, 1, 7680)
    std = np.std(trail_processing_data, axis=(0, 1, 2), keepdims=True)

    trail_processing_data = (trail_processing_data - mean) / std

    return trail_processing_data


def other_final_proccessing(other_trail_data):
    """
    将其他生理信号切割成 60 段并进行拼接
    :param other_trail_data:
    :return:
    """
    assert other_trail_data == (40, 4, 7680)

    final_other_datas = np.array_split(other_trail_data, 60, axis=2)

    final_other_data = np.concatenate(final_other_datas, axis=0)

    assert final_other_data == (2400, 4, 128)

    return final_other_data


def eeg_final_proccessing(mapped_data):
    """
    将EEG信号数据进行最终的处理， 得到(40*60, 9, 9, 128)结构的数据
    :param mapped_data: 映射为二维矩阵后的数据
    :return:
    """
    mapped_datas = np.array_split(mapped_data, 60, axis=3)
    assert mapped_datas[0].shape == (40, 9, 9, 128)

    final_eeg_data = np.concatenate(mapped_datas, axis=0)

    assert final_eeg_data.shape == (2400, 9, 9, 128)  # (40trails * 60s, 9, 9, 128)

    return final_eeg_data


def eeg_final_proccessing2(mapped_data):
    """
    将EEG信号数据进行最终的处理， 得到(40*10, 9, 9,768)结构的数据
    :param mapped_data: 映射为二维矩阵后的数据
    :return:
    """
    mapped_datas = np.array_split(mapped_data, 10, axis=3)
    assert mapped_datas[0].shape == (40, 9, 9, 768)

    final_eeg_data = np.concatenate(mapped_datas, axis=0)

    assert final_eeg_data.shape == (400, 9, 9, 768)  # (40trails * 60s, 9, 9, 128)

    return final_eeg_data


def other_final_proccessing(other_trail_data):
    """
    将其他信号进行最终的处理， 将60s分段， 得到(40*60, 4, 128)的数据
    :param other_trail_data:
    :return:
    """

    assert other_trail_data.shape == (40, 4, 7680)
    other_final_datas = np.array_split(other_trail_data, 60, axis=2)

    final_other_data = np.concatenate(other_final_datas, axis=0)

    assert final_other_data.shape == (2400, 4, 128)

    return final_other_data

def eeg_final_processing3(eeg_trail_data):

    assert eeg_trail_data.shape == (40, 32, 7680)
    eeg_final_datas = np.array_split(eeg_trail_data, 60, axis=2)

    final_eeg_data = np.concatenate(eeg_final_datas, axis=0)

    assert final_eeg_data.shape == (2400, 32, 128)

    return final_eeg_data


def map_matrix(trail_processing_data):
    """
    将通道位置映射到二维数组上
    :param trail_processing_data:
    :return: 40实验 * 9 * 9的二维矩阵 * 7680个数据点
    """
    electrode_positions = np.array([  # 定义电极位置
        [0, 3],
        [1, 3],
        [2, 2],
        [2, 0],
        [3, 1],
        [3, 3],
        [4, 2],
        [4, 0],
        [5, 1],
        [5, 3],
        [6, 2],
        [6, 0],
        [7, 3],
        [8, 3],
        [8, 4],
        [6, 4],
        [0, 5],
        [1, 5],
        [2, 4],
        [2, 6],
        [2, 8],
        [3, 7],
        [3, 5],
        [4, 4],
        [4, 6],
        [4, 8],
        [5, 7],
        [5, 5],
        [6, 6],
        [6, 8],
        [7, 5],
        [8, 5]
    ])
    mapped_data = np.zeros((40, 9, 9, 7680))
    for i in range(40):
        for j in range(32):
            position = electrode_positions[j]
            mapped_data[i, position[0], position[1], :] = trail_processing_data[i, j, :]

    assert mapped_data.shape == (40, 9, 9, 7680)
    return mapped_data


def proccessing_labels(labels):
    """
    预处理标签数组
    :param labels:
    :return:
    """
    new_labels = labels[:, :2]
    new_labels = np.tile(new_labels, (60, 1))
    assert new_labels.shape == (2400, 2)

    # 第一个是valence， 第二个是arousal
    return new_labels[:, 0], new_labels[:, 1]


def proccessing_labels_2(labels):
    """
    预处理标签数组
    :param labels:
    :return:
    """
    new_labels = labels[:, :2]
    new_labels = np.tile(new_labels, (6, 1))
    assert new_labels.shape == (240, 2)

    # 第一个是valence， 第二个是arousal
    return new_labels[:, 0], new_labels[:, 1]


def data_proccessing2(filepath):
    """

    :param filepath:
    :return:
    """

    deap = DEAP(filepath)

    subject_num = deap.__len__()

    data_list = []
    eeg_array_data = []
    other_array_data = []
    v_array_labels = []
    a_array_labels = []

    for i in range(subject_num):
        eeg_baseline_data, other_baseline_data, eeg_trail_data, other_trail_data, labels = deap.__getitem__(i)
        eeg_baseline_mean = baseline_sigal_processing(eeg_baseline_data)  # eeg基线数据做平均
        # other_baseline_mean = baseline_sigal_processing(other_baseline_data)  # 其他信号基线数据做平均

        eeg_trail_procceesing_data = trail_signal_proccessing(eeg_trail_data, eeg_baseline_mean)
        # other_trail_proccessing_data = trail_signal_proccessing(other_trail_data, other_baseline_mean)

        mapped_data = map_matrix(eeg_trail_procceesing_data)

        eeg_final_data = eeg_final_proccessing2(mapped_data)
        # other_final_data = other_final_proccessing(other_trail_proccessing_data)

        v_labels, a_labels = proccessing_labels_2(labels)

        eeg_array_data.append(eeg_final_data)
        # other_array_data.append(other_final_data)

        v_array_labels.append(v_labels)
        a_array_labels.append(a_labels)

    EEG = np.concatenate(eeg_array_data, axis=0)
    assert EEG.shape == (12800, 9, 9, 768)

    EEG = torch.FloatTensor(EEG)

    v_labels = np.concatenate(v_array_labels)
    assert v_labels.shape == (12800,)
    a_labels = np.concatenate(a_array_labels)

    return EEG, v_labels, a_labels

def data_proccessing3(filepath):
    """
    :param filepath:
    :return:
    """
    deap = DEAP(filepath)
    subject_num = deap.__len__()

    data_list = []
    eeg_array_data = []
    other_array_data = []
    v_array_labels = []
    a_array_labels = []

    for i in range(subject_num):
        eeg_baseline_data, other_baseline_data, eeg_trail_data, other_trail_data, labels = deap.__getitem__(i)
        eeg_baseline_mean = baseline_sigal_processing(eeg_baseline_data)  # eeg基线数据做平均
        other_baseline_mean = baseline_sigal_processing(other_baseline_data)  # 其他信号基线数据做平均

        eeg_trail_procceesing_data = trail_signal_proccessing(eeg_trail_data, eeg_baseline_mean)
        other_trail_proccessing_data = trail_signal_proccessing(other_trail_data, other_baseline_mean)


        eeg_final_data = eeg_final_processing3(eeg_trail_procceesing_data)
        other_final_data = other_final_proccessing(other_trail_proccessing_data)

        v_labels, a_labels = proccessing_labels(labels)

        eeg_array_data.append(eeg_final_data)
        other_array_data.append(other_final_data)

        v_array_labels.append(v_labels)
        a_array_labels.append(a_labels)

    EEG = np.concatenate(eeg_array_data, axis=0)
    other_data = np.concatenate(other_array_data, axis=0)
    assert EEG.shape == (76800, 32, 128)
    EOG = other_data[:, 0:2, :]
    EMG = other_data[:, 2:4, :]
    assert EOG.shape == (76800, 2, 128)
    assert EMG.shape == (76800, 2, 128)

    EEG = torch.FloatTensor(EEG)
    EOG = torch.FloatTensor(EOG)
    EMG = torch.FloatTensor(EMG)

    v_labels = np.concatenate(v_array_labels)
    assert v_labels.shape == (76800,)
    a_labels = np.concatenate(a_array_labels)

    return EEG, EOG, EMG, v_labels, a_labels


def data_proccessing(filepath):
    """

    :param filepath:
    :return:
    """
    deap = DEAP(filepath)

    subject_num = deap.__len__()

    data_list = []
    eeg_array_data = []
    other_array_data = []
    v_array_labels = []
    a_array_labels = []

    for i in range(subject_num):
        eeg_baseline_data, other_baseline_data, eeg_trail_data, other_trail_data, labels = deap.__getitem__(i)
        eeg_baseline_mean = baseline_sigal_processing(eeg_baseline_data)  # eeg基线数据做平均
        other_baseline_mean = baseline_sigal_processing(other_baseline_data)  # 其他信号基线数据做平均

        eeg_trail_procceesing_data = trail_signal_proccessing(eeg_trail_data, eeg_baseline_mean)
        other_trail_proccessing_data = trail_signal_proccessing(other_trail_data, other_baseline_mean)

        # mapped_data = map_matrix(eeg_trail_procceesing_data)

        # eeg_final_data = eeg_final_proccessing(mapped_data)
        eeg_final_data = eeg_final_processing3(eeg_trail_procceesing_data)
        other_final_data = other_final_proccessing(other_trail_proccessing_data)

        v_labels, a_labels = proccessing_labels(labels)

        eeg_array_data.append(eeg_final_data)
        other_array_data.append(other_final_data)

        v_array_labels.append(v_labels)
        a_array_labels.append(a_labels)

    other_data = np.concatenate(other_array_data, axis=0)
    EEG = np.concatenate(eeg_array_data, axis=0)

    hEOG = other_data[:, :1, :]
    vEOG = other_data[:, 1:2, :]
    zEMG = other_data[:, 2:3, :]
    tEMG = other_data[:, 3:4, :]

    EEG = torch.FloatTensor(EEG)
    hEOG = torch.FloatTensor(hEOG)
    vEOG = torch.FloatTensor(vEOG)
    zEMG = torch.FloatTensor(zEMG)
    tEMG = torch.FloatTensor(tEMG)

    v_labels = np.concatenate(v_array_labels)
    a_labels = np.concatenate(a_array_labels)

    assert EEG.shape == (76800, 32, 128)
    assert v_labels.shape == (76800,)

    return EEG, hEOG, vEOG, zEMG, tEMG, v_labels, a_labels


if __name__ == '__main__':
    filepath = '..\\data\\data_preprocessed_matlab'

    data_proccessing2(filepath)
