# -*- coding: utf-8 -*-
"""
UNI 包装类（来自 MahmoodLab/CLAM）。
- 输入: 224×224 RGB 图像 (已归一化)
- 输出: (logits [2分类], features [1024-dim])
- 依赖: transformers (ViTModel), torchvision
"""

import os
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class UNI(nn.Module):
    """
    包装 UNI (ViT-L/16) 模型。
    - 官方 HF 仓库: https://huggingface.co/MahmoodLab/UNI
    - 特征维度: 1024
    - 典型用法: 作为 CLAM 等弱监督学习框架的预训练编码器
    """

    def __init__(self, ckpt_path=None):
        """
        :param ckpt_path: 可选的本地 checkpoint (.pth) 路径。
                      如果不提供，将使用 HuggingFace 上的预训练权重。
        """
        super().__init__()

        # 加载 ViT 配置与主干
        self.config = ViTConfig.from_pretrained(UNI_MODEL_NAME)
        self.backbone = ViTModel.from_pretrained(UNI_MODEL_NAME)
        self.backbone.eval()

        # 可选：加载自定义 checkpoint（如果有的话）
        if ckpt_path and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            # 检查状态键是否匹配，这里做简单的 strict=False 兼容
            self.backbone.load_state_dict(state, strict=False)

        # 添加一个简单的分类头（二分类），以便在 forward 时返回 logits
        self.fc = nn.Linear(self.backbone.config.hidden_size, 2)
        self.fc.weight.data.normal_(mean=0.0, std=0.02)
        self.fc.bias.data.zero_()

        # 特征维度（CLAM 文档中 UNI 输出 1024-dim）
        self._feat_dim = self.backbone.config.hidden_size  # == 1024

    def forward(self, x):
        """
        :param x: Tensor [B, 3, 224, 224]，已经按照 [0.485,0.456,0.406] / [0.229,0.224,0.225] 归一化
        :return: (logits [B, 2], features [B, 1024])
        """
        # ViT 的 forward 返回 BaseModelOutputWithLastHiddenState
        output = self.backbone(pixel_values=x)
        # 取 [CLS] token 作为整个图像的表示
        pooled = output.last_hidden_state[:, 0]  # [B, hidden_dim]
        logits = self.fc(pooled)                # [B, 2]
        return logits, pooled

    def extract_features(self, x):
        """
        只返回特征向量（不包括分类 logits），便于在 CLAM/其他流程中使用。
        :param x: [B, 3, 224, 224]
        :return: features [B, 1024]
        """
        with torch.no_grad():
            logits, feat = self.forward(x)
        return feat

    @property
    def feature_dim(self):
        """返回特征维度，便于在其他模块中动态对齐。"""
        return self._feat_dim


# 方便直接 import 使用：创建一个默认实例（不带 checkpoint）
# 用户如有本地权重，可改为: uni = UNI(ckpt_path="/path/to/pytorch_model.bin")
_default_uni = UNI()
_default_uni.eval()


def get_uni():
    """返回默认的 UNI 实例，供其他模块直接使用。"""
    return _default_uni