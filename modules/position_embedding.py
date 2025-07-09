import math
import torch
import torch.nn as nn

# Code adapted from the fairseq repo.

def make_positions(tensor, padding_idx, left_pad):
    """
    这个函数主要是用来将一个序列中的非填充符号替换为它们在序列中的位置编号。
    序列中的填充符号会被忽略，并且支持左填充和右填充两种方式。
    当 left_pad=True 时，填充符号位于序列的左边，位置编号会相应调整。
    Args:
        tensor:
        padding_idx:
        left_pad:

    Returns:

    """
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    将非填充符号替换为它们的位置编号。
    位置编号从 padding_idx + 1 开始。
    填充符号会被忽略，但需要指定填充是添加在左侧（left_pad=True）还是右侧（left_pad=False）
    """
    # 计算最大位置索引 tensor.size(1):tensor 的第二维度的大小
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    #根据设备名称创建一个缓冲区的名字 buf_name，格式为 'range_buf_device_id'
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    这个模块生成任意长度的正弦位置嵌入。
    填充符号会被忽略，但需要指定填充是添加在左侧（left_pad=True）还是右侧（left_pad=False）
    """
    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()   # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

if __name__ == '__main__':
    tensor = torch.tensor([
        [1, 3, 3, 9, 9, 10, 2],
        [1, 99, 4, 0, 7, 9, 99],
        [100, 200, 3, 100, 786, 200, 1]
    ])
    # print(make_positions(tensor, 1, False))
    # tensor([[1, 3, 4, 5, 6, 7, 8],
    #         [1, 3, 4, 5, 6, 7, 8],
    #         [2, 3, 4, 5, 6, 7, 1]])
