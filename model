import torch
from torch import nn
from monai.networks.nets import EfficientNetBN


class PrognosisModelD(nn.Module):
    def __init__(self):
        super(PrognosisModelD, self).__init__()
        self.extractor_liver = EfficientNetBN(model_name="efficientnet-b0", in_channels=1, spatial_dims=2, num_classes=32)
        self.extractor_lung = EfficientNetBN(model_name="efficientnet-b0", in_channels=1, spatial_dims=2, num_classes=32)

        self.LSTM_liver = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True)
        self.LSTM_lung = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True)

        self.cls_1 = nn.Linear(64, 8)
        self.cls_2 = nn.Linear(8, 1)
        self.drop = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, x_liver, x_lung, x_liver_1, x_lung_1):
        liver_feat, lung_feat, liver_feat_1, lung_feat_1 = [], [], [], []
        for idx in range(3):
            x_liver_out = self.extractor_liver(x_liver[:, [idx], :, :])
            liver_feat.append(x_liver_out)
            x_lung_out = self.extractor_lung(x_lung[:, [idx], :, :])
            lung_feat.append(x_lung_out)
            x_liver_out_1 = self.extractor_liver(x_liver_1[:, [idx], :, :])
            liver_feat_1.append(x_liver_out_1)
            x_lung_out_1 = self.extractor_lung(x_lung_1[:, [idx], :, :])
            lung_feat_1.append(x_lung_out_1)
        x_liver = torch.mean(torch.stack(liver_feat), dim=0)
        x_lung = torch.mean(torch.stack(lung_feat), dim=0)
        x_liver_1 = torch.mean(torch.stack(liver_feat_1), dim=0)
        x_lung_1 = torch.mean(torch.stack(lung_feat_1), dim=0)

        self.LSTM_liver.flatten_parameters()
        self.LSTM_lung.flatten_parameters()
        liver_out, (h_n, h_c) = self.LSTM_liver(torch.stack([x_liver, x_liver_1], dim=1), None)
        lung_out, (h_n, h_c) = self.LSTM_lung(torch.stack([x_lung, x_lung_1], dim=1), None)

        x_liver = liver_out[:, -1, :]
        x_lung = lung_out[:, -1, :]

        out = torch.concat([x_liver, x_lung], dim=1)
        out = self.cls_1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.cls_2(out)

        return out


class PrognosisModelS(nn.Module):
    def __init__(self):
        super(PrognosisModelS, self).__init__()
        self.extractor_liver = EfficientNetBN(model_name="efficientnet-b0", in_channels=1, spatial_dims=2, num_classes=32)
        self.extractor_lung = EfficientNetBN(model_name="efficientnet-b0", in_channels=1, spatial_dims=2, num_classes=32)

        self.cls_1 = nn.Linear(64, 8)
        self.cls_2 = nn.Linear(8, 1)
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

    def forward(self, x_liver, x_lung):
        liver_feat, lung_feat = [], []
        for idx in range(3):
            x_liver_out = self.extractor_liver(x_liver[:, [idx], :, :])
            liver_feat.append(x_liver_out)
            x_lung_out = self.extractor_lung(x_lung[:, [idx], :, :])
            lung_feat.append(x_lung_out)
        x_liver = torch.mean(torch.stack(liver_feat), dim=0)
        x_lung = torch.mean(torch.stack(lung_feat), dim=0)

        out = torch.concat([x_liver, x_lung], dim=1)
        out = self.cls_1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.cls_2(out)

        return out
