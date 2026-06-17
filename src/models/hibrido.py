import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalCNN(nn.Module):
    def __init__(
        self,
        in_channels=18,
        window_size=256,
        temporal_filters=16,
        spatial_filters=32,
        num_classes=1,
        temporal_kernel=15,
        dropout_p=0.3,
        spatial_dropout_p=0.1,
        temporal_norm='instance',
    ):
        super().__init__()
        assert temporal_kernel % 2 == 1, 'temporal_kernel debe ser impar.'

        self.in_channels = in_channels
        self.window_size = window_size
        self.temporal_filters = temporal_filters
        self.spatial_filters = spatial_filters

        self.temporal_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * temporal_filters,
            kernel_size=temporal_kernel,
            padding=(temporal_kernel - 1) // 2,
            groups=in_channels,
            bias=False,
        )
        self.temporal_shortcut = nn.Conv1d(
            in_channels,
            in_channels * temporal_filters,
            kernel_size=1,
            groups=in_channels,
            bias=False,
        )

        if temporal_norm == 'instance':
            self.bn_temp = nn.InstanceNorm1d(in_channels * temporal_filters, affine=True)
        elif temporal_norm == 'batch':
            self.bn_temp = nn.BatchNorm1d(in_channels * temporal_filters)
        else:
            raise ValueError("temporal_norm debe ser 'instance' o 'batch'.")

        self.spatial_conv = nn.Conv1d(
            in_channels=in_channels * temporal_filters,
            out_channels=spatial_filters,
            kernel_size=1,
            bias=False,
        )
        self.spatial_shortcut = nn.Conv1d(
            in_channels * temporal_filters,
            spatial_filters,
            kernel_size=1,
            bias=False,
        )
        self.bn_spat = nn.InstanceNorm1d(spatial_filters, affine=True)

        self.pool = nn.AvgPool1d(kernel_size=8, stride=8)
        self.spatial_dropout = nn.Dropout1d(p=spatial_dropout_p)

        self.conv_refine = nn.Conv1d(
            spatial_filters,
            spatial_filters * 2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.refine_shortcut = nn.Conv1d(
            spatial_filters,
            spatial_filters * 2,
            kernel_size=1,
            bias=False,
        )
        self.bn_refine = nn.InstanceNorm1d(spatial_filters * 2, affine=True)
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(p=dropout_p)

        self.final_feature_dim = self._infer_feature_dim()
        self.fc = nn.Linear(self.final_feature_dim, num_classes)

    def _forward_features(self, x):
        residual1 = self.temporal_shortcut(x)
        x = self.temporal_conv(x)
        x = self.bn_temp(x)
        x = F.elu(x + residual1)

        residual2 = self.spatial_shortcut(x)
        x = self.spatial_conv(x)
        x = self.bn_spat(x)
        x = F.elu(x + residual2)

        x = self.pool(x)
        x = self.spatial_dropout(x)

        residual3 = self.refine_shortcut(x)
        x = self.conv_refine(x)
        x = self.bn_refine(x)
        x = F.elu(x + residual3)

        x = self.pool2(x)
        x = self.spatial_dropout(x)
        return x

    def _infer_feature_dim(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.window_size)
            x = self._forward_features(dummy)
            feature_dim = x.flatten(1).shape[1]
        self.train(was_training)
        return feature_dim

    def forward(self, x, return_embedding=False, return_logits=True):
        x = self._forward_features(x)
        x = x.flatten(1)
        if return_embedding and not return_logits:
            return x
        x = self.dropout(x)
        logits = self.fc(x)
        if return_embedding:
            return logits, x
        return logits

    def freeze_temporal_block(self, freeze_shortcut=True, freeze_norm=True):
        for p in self.temporal_conv.parameters():
            p.requires_grad = False
        if freeze_shortcut:
            for p in self.temporal_shortcut.parameters():
                p.requires_grad = False
        if freeze_norm:
            for p in self.bn_temp.parameters():
                p.requires_grad = False

    def unfreeze_temporal_block(self):
        for module in [self.temporal_conv, self.temporal_shortcut, self.bn_temp]:
            for p in module.parameters():
                p.requires_grad = True

    def freeze_backbone_except_head(self):
        for module in [
            self.temporal_conv,
            self.temporal_shortcut,
            self.bn_temp,
            self.spatial_conv,
            self.spatial_shortcut,
            self.bn_spat,
            self.conv_refine,
            self.refine_shortcut,
            self.bn_refine,
        ]:
            for p in module.parameters():
                p.requires_grad = False
        for p in self.fc.parameters():
            p.requires_grad = True

    def get_spatial_attention(self, x_after_spatial):
        return torch.softmax(x_after_spatial.mean(dim=-1), dim=1)