# -*- coding: utf-8 -*-       
# __@Time__    : 2025-07-17 15:47   
# __@Author__  : www             
# __@File__    : channel_attention.py        
# __@Description__ :
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
class channel_attention(nn.Module):
    """
    通道注意力
    """
    def __init__(self, sequence_num=128, inter=4, channel_num=32):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.channel_num = channel_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(self.channel_num, self.channel_num),
            nn.LayerNorm(self.channel_num),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(self.channel_num, self.channel_num),
            # nn.LeakyReLU(),
            nn.LayerNorm(self.channel_num),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(self.channel_num, self.channel_num),
            # nn.LeakyReLU(),
            nn.LayerNorm(self.channel_num),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out
