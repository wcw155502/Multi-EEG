from torch import nn
import torch
from torch import Tensor
from modules.transformer import TransformerEncoder
from einops.layers.torch import Rearrange


class EEGEncoder(nn.Module):
    def __init__(self, embed_size: int = 40, kernal=32):
        super(EEGEncoder, self).__init__()
        self.t_conv2d = nn.Conv2d(1, 40, (1, 16), (1, 1))
        self.s_conv2d = nn.Conv2d(40, 40, (kernal, 1), (1, 1))
        self.bn = nn.BatchNorm2d(40)
        self.elu = nn.ELU()
        self.avgpool = nn.AvgPool2d((1, 32), (1, 12))
        self.dropout = nn.Dropout(0.5)

        self.projection = nn.Sequential(
            nn.Conv2d(40, embed_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.t_conv2d(x)
        x = self.s_conv2d(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x


class OtherSingalEncoder(nn.Module):
    def __init__(self, embed_size: int = 40):
        super(OtherSingalEncoder, self).__init__()
        self.conv = nn.Conv1d(1, 40, 16, 1)
        self.bn = nn.BatchNorm2d(40)
        self.elu = nn.ELU()
        self.avgpool = nn.AvgPool1d(32, 12)
        self.dropout = nn.Dropout(0.5)

        self.projection = nn.Sequential(
            nn.Conv1d(40, embed_size, 1, stride=1),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x


class Cross_TransModel1(nn.Module):
    """
    交叉注意力模型
    """

    def __init__(self):
        super(Cross_TransModel, self).__init__()
        self.d_m = 40
        self.num_heads = 4
        self.layers = 3
        self.attn_dropout = 0.05
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.1
        self.embed_dropout = 0.1
        self.attn_mask = True
        self.out_classes = 2

        # 时间卷积层
        self.eeg_conv = EEGEncoder(self.d_m, 32)
        self.eog_conv = EEGEncoder(self.d_m, 2)
        self.emg_conv = EEGEncoder(self.d_m, 2)

        # 交叉注意力
        # self.trans_eeg = self.get_network()
        # self.trans_eog = self.get_network()
        # self.trans_emg = self.get_network()
        # 为每对模态定义交叉注意力模块
        self.eeg2eog = self.get_network()
        self.eog2emg = self.get_network()
        self.emg2eeg = self.get_network()

        # 自适应特征融合权重
        self.alpha = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]))

        # 自注意力
        self.eeg_self = self.get_network()
        self.eog_self = self.get_network()
        self.emg_self = self.get_network()

        # 预测层
        # self.final_conv = ConLayer(36, 1)
        # self.final_conv2 = ConLayer(16, 1)
        # self.fc = nn.Linear(280, 32)
        # self.out_layer = nn.Linear(64, self.out_classes)
        self.fc = nn.Sequential(
            nn.Linear(840, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.out_classes)
        )

        self.dropout = nn.Dropout(0.5)

    def get_network(self):
        return TransformerEncoder(embed_dim=40,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask
                                  )

    def forward(self, eeg, eog, emg):
        # (batch, src_len, embed_dim)
        # eeg = eeg.transpose(1, 2)
        # eog = eog.transpose(1, 2)
        # emg = emg.transpose(1, 2)
        #
        # eeg = self.eeg_conv(eeg)
        # eog = self.eog_conv(eog)
        # emg = self.emg_conv(emg)
        #
        # eeg = eeg.transpose(0, 1)# (src_len, batch, embed_dim) ->(通道数, batch, 采样率)
        # eog = eog.transpose(0, 1)
        # emg = emg.transpose(0, 1)

        # all_features = torch.cat([eeg, eog, emg], dim=0)

        # eeg_with_all = self.trans_eeg(eeg, all_features)
        # eog_with_all = self.trans_eog(eog, all_features)
        # emg_with_all = self.trans_emg(emg, all_features)
        eeg2eog = self.eeg2eog(eeg, eog)
        eog2emg = self.eog2emg(eog, emg)
        emg2eeg = self.emg2eeg(emg, eeg)

        self_eeg = self.eeg_self(eeg, eeg)
        self_eog = self.eog_self(eog, eog)
        self_emg = self.emg_self(emg, emg)

        eeg_out = self.alpha[0] * self_eeg + self.alpha[1] * eeg2eog
        eog_out = self.alpha[1] * self_eog + self.alpha[2] * eog2emg
        emg_out = self.alpha[2] * self_emg + self.alpha[0] * emg2eeg

        return eeg_out, eog_out, emg_out

        # out_all = torch.cat([eeg_with_eeg, eeg_with_all, eog_with_all, emg_with_all], dim=0)
        # out_all = torch.cat([eeg_with_eeg, eeg_with_eog, eeg_with_emg], dim=0)
        # out = out_all.permute(1, 0, 2)
        # out = out.contiguous().view(out.size(0), -1)
        # # out = self.final_conv(out_all.permute(1, 0, 2)).squeeze(1)
        # out = self.fc(out)


class Cross_TransModel(nn.Module):
    """
    交叉注意力模型
    """

    def __init__(self):
        super(Cross_TransModel, self).__init__()
        self.d_m = 40
        self.num_heads = 4
        self.layers = 3
        self.attn_dropout = 0.05
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.1
        self.embed_dropout = 0.1
        self.attn_mask = True
        self.out_classes = 2

        # 时间卷积层
        self.eeg_conv = EEGEncoder(self.d_m, 32)
        self.eog_conv = OtherSingalEncoder(self.d_m)
        self.eog_conv = OtherSingalEncoder(self.d_m)

        # 交叉注意力
        self.trans_eeg = self.get_network()
        self.trans_eog = self.get_network()
        self.trans_emg = self.get_network()

        # 预测层
        self.fc = nn.Sequential(
            nn.Linear(840, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.out_classes)
        )

        self.dropout = nn.Dropout(0.5)

    def get_network(self):
        return TransformerEncoder(embed_dim=40,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask
                                  )

    def forward(self, eeg, eog1, eog2, emg1, emg2):
        # (batch, src_len, embed_dim)
        eeg = self.eeg_conv(eeg)
        # eog = self.eog_conv(eog)
        # emg = self.emg_conv(emg)
        eog1 = self.eog_conv(eog1)
        emg1 = self.emg_conv(emg1)
        eog2 = self.eog_conv(eog2)
        emg2 = self.emg_conv(emg2)

        eeg = eeg.transpose(0, 1)  # (src_len, batch, embed_dim) ->(通道数, batch, 采样率)
        # eog = eog.transpose(0, 1)
        # emg = emg.transpose(0, 1)
        eog1 = eog1.transpose(0, 1)
        emg1 = emg1.transpose(0, 1)
        eog2 = eog2.transpose(0, 1)
        emg2 = emg2.transpose(0, 1)

        eog = torch.cat([eog1, eog2], dim=0)
        emg = torch.cat([emg1, emg2], dim=0)

        all_features = torch.cat([eeg, eog, emg], dim=0)

        eeg_with_all = self.trans_eeg(eeg, all_features)
        eog_with_all = self.trans_eog(eog, all_features)
        emg_with_all = self.trans_emg(emg, all_features)

        out_all = torch.cat([eeg_with_all, eog_with_all, emg_with_all], dim=0)
        # out_all = torch.cat([eeg_with_eeg, eeg_with_eog, eeg_with_emg], dim=0)
        out = out_all.permute(1, 0, 2)
        out = out.contiguous().view(out.size(0), -1)
        # out = self.final_conv(out_all.permute(1, 0, 2)).squeeze(1)
        out = self.fc(out)
        return out

# class TransModel(nn.Module):
#
#     def __init__(self):
#         super(TransModel, self).__init__()
#
#         self.d_m = 40
#         self.out_classes = 2
#
#         # 时间卷积层
#         self.eeg_conv = EEGEncoder(self.d_m, 32)
#         self.eog_conv = EEGEncoder(self.d_m, 2)
#         self.emg_conv = EEGEncoder(self.d_m, 2)
#
#         self.fusion_layers = nn.ModuleList([
#             Cross_TransModel()
#             for _ in range(3)
#         ])
#
#         self.fc = nn.Sequential(
#             nn.Linear(840, 256),
#             nn.ELU(),
#             nn.Dropout(0.4),
#             nn.Linear(256, self.out_classes)
#         )
#
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, eeg, eog, emg):
#         eeg = self.eeg_conv(eeg)
#         eog = self.eog_conv(eog)
#         emg = self.emg_conv(emg)
#
#         eeg = eeg.transpose(0, 1)# (src_len, batch, embed_dim) ->(通道数, batch, 采样率)
#         eog = eog.transpose(0, 1)
#         emg = emg.transpose(0, 1)
#
#         for layer in self.fusion_layers:
#             eeg_feat, eog_feat, emg_feat = layer(eeg, eog, emg)
#
#         out_all = torch.cat([eeg_feat, eog_feat, emg_feat], dim=0)
#         out = out_all.permute(1, 0, 2)
#         out = out.contiguous().view(out.size(0), -1)
#         # out = self.final_conv(out_all.permute(1, 0, 2)).squeeze(1)
#         out = self.fc(out)
#
#         return out
