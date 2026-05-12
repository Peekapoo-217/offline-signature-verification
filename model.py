import torch
import torch.nn as nn
from torchvision.models import resnet18

from config import EMBED_DIM


class SiameseNetwork(nn.Module):
    """
    Siamese Network: Hai ảnh đi qua CÙNG MỘT backbone (weight sharing)
    để tạo ra 2 embedding vectors.
    
    Cấu trúc thực sự của model trong best_model.pth là ResNet18 (được modify 
    nhận ảnh 1 channel grayscale), theo sau là custom Neck (Linear -> BN -> ReLU -> Dropout -> Linear).
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super(SiameseNetwork, self).__init__()

        # Khởi tạo ResNet18
        resnet = resnet18(weights=None)
        
        # Modify layer đầu tiên để nhận input 1-channel (grayscale) thay vì 3-channel (RGB)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Backbone là toàn bộ ResNet18 NGOẠI TRỪ lớp Linear(fc) cuối cùng.
        # Lớp AdaptiveAvgPool2d (layer số 8) vẫn được giữ lại.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Neck section matches the checkpoint's missing parameters (Linear -> BN -> Linear)
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Dropout probability parameterly matched with general use.
            nn.Linear(512, embed_dim)
        )

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Path 1
        x1 = self.backbone(img1)
        embed1 = self.neck(x1)
        
        # Path 2
        x2 = self.backbone(img2)
        embed2 = self.neck(x2)
        
        return embed1, embed2
