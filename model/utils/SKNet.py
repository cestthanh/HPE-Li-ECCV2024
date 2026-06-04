# -*- coding: utf-8 -*-

import torch
from torch import nn

from .utils import ScaledDotProductAttention


class SKConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dim1,
        dim2,
        pool_dim,
        M=4,
        G=1,
        r=4,
        stride=1,
        L=32,
        use_min_l=False,
        layout="legacy_view",
    ):
        """Selective-kernel convolution with optional legacy checkpoint behavior.

        `legacy_view` preserves the historical implementation, including its
        channel/frequency memory reinterpretation. `stack` keeps the branch,
        channel, frequency, and time axes explicit.
        """
        super(SKConv, self).__init__()

        if layout not in {"legacy_view", "stack"}:
            raise ValueError(f"Unsupported SKConv layout: {layout}")
        if input_dim % G != 0 or output_dim % G != 0:
            raise ValueError(
                f"SKConv groups G={G} must divide input_dim={input_dim} "
                f"and output_dim={output_dim}."
            )

        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.M = M
        self.pool_dim = pool_dim
        self.layout = layout
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_dim,
                        output_dim,
                        kernel_size=3,
                        stride=stride,
                        padding=1 + i,
                        dilation=1 + i,
                        groups=G,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True),
                )
                for i in range(M)
            ]
        )

        def bottleneck_dim(source_dim):
            d = int(source_dim / r)
            if use_min_l:
                d = max(d, L)
            if d <= 0:
                raise ValueError(
                    f"Invalid SKConv bottleneck d={d} from source_dim={source_dim}, r={r}."
                )
            return d

        if pool_dim == "freq":
            d = bottleneck_dim(dim1)
            self.fc = nn.Sequential(
                nn.Linear(dim1, d),
                nn.BatchNorm1d(d),
                nn.ReLU(inplace=True),
            )
            self.fcs = nn.ModuleList(
                [
                    nn.Conv1d(d, dim1, kernel_size=1, stride=1)
                    for _ in range(M)
                ]
            )
        elif pool_dim == "freq-time":
            d = bottleneck_dim(dim1 * dim2)
            self.fc = nn.Sequential(
                nn.Linear(dim1 * dim2, d),
                nn.BatchNorm1d(d),
                nn.ReLU(inplace=True),
            )
            self.fcs = nn.ModuleList(
                [
                    nn.Conv1d(d, dim1 * dim2, kernel_size=1, stride=1)
                    for _ in range(M)
                ]
            )
        elif pool_dim == "freq-chan":
            d = bottleneck_dim(output_dim)
            self.fc = nn.Sequential(
                nn.Conv1d(output_dim, d, kernel_size=1, stride=1),
                nn.BatchNorm1d(d),
                nn.ReLU(inplace=True),
            )
            self.fcs = nn.ModuleList(
                [
                    nn.Conv1d(d, output_dim, kernel_size=1, stride=1)
                    for _ in range(M)
                ]
            )
        else:
            raise ValueError(f"Unsupported SKConv pool_dim: {pool_dim}")

        self.softmax = nn.Softmax(dim=1)

    def _forward_legacy_view(self, x):
        batch_size = x.shape[0]
        feats = torch.cat([conv(x) for conv in self.convs], dim=1)
        feats = feats.view(
            batch_size,
            self.M,
            feats.shape[2],
            self.output_dim,
            feats.shape[3],
        )
        feats_u = torch.sum(feats, dim=1)

        if self.pool_dim == "freq":
            feats_s = torch.mean(feats_u, dim=[2, 3])
            feats_z = self.fc(feats_s).unsqueeze(2)
            attention = torch.cat([fc(feats_z) for fc in self.fcs], dim=1)
            attention = attention.view(batch_size, self.M, self.dim1, 1, 1)
        elif self.pool_dim == "freq-time":
            feats_s = torch.mean(feats_u, dim=2)
            feats_s = feats_s.view(batch_size, feats_s.shape[1] * feats_s.shape[2])
            feats_z = self.fc(feats_s).unsqueeze(2)
            attention = torch.cat([fc(feats_z) for fc in self.fcs], dim=1)
            attention = attention.view(
                batch_size, self.M, self.dim1 * self.dim2, 1, 1
            )
            attention = self.softmax(attention)
            attention = attention.view(
                batch_size, self.M, self.dim1, 1, self.dim2
            )
            return torch.transpose(torch.sum(feats * attention, dim=1), 1, 2)
        else:
            feats_s = torch.mean(feats_u, dim=3)
            feats_s = feats_s.view(batch_size, feats_s.shape[2], feats_s.shape[1])
            feats_z = self.fc(feats_s)
            attention = torch.cat([fc(feats_z) for fc in self.fcs], dim=1)
            attention = attention.view(
                batch_size, self.M, self.output_dim, self.dim1, 1
            )
            attention = self.softmax(attention)
            attention = attention.view(
                batch_size, self.M, self.dim1, self.output_dim, 1
            )
            return torch.transpose(torch.sum(feats * attention, dim=1), 1, 2)

        attention = self.softmax(attention)
        return torch.transpose(torch.sum(feats * attention, dim=1), 1, 2)

    def _forward_stack(self, x):
        # [B, M, C, F, T], without reinterpreting channel/frequency memory.
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)
        feats_u = torch.sum(feats, dim=1)

        if self.pool_dim == "freq":
            feats_s = torch.mean(feats_u, dim=(1, 3))
            feats_z = self.fc(feats_s).unsqueeze(2)
            attention = torch.stack([fc(feats_z).squeeze(2) for fc in self.fcs], dim=1)
            attention = self.softmax(attention)[:, :, None, :, None]
        elif self.pool_dim == "freq-time":
            feats_s = torch.mean(feats_u, dim=1).reshape(x.shape[0], -1)
            feats_z = self.fc(feats_s).unsqueeze(2)
            attention = torch.stack([fc(feats_z).squeeze(2) for fc in self.fcs], dim=1)
            attention = self.softmax(attention).view(
                x.shape[0], self.M, 1, self.dim1, self.dim2
            )
        else:
            feats_s = torch.mean(feats_u, dim=3)
            feats_z = self.fc(feats_s)
            attention = torch.stack([fc(feats_z) for fc in self.fcs], dim=1)
            attention = self.softmax(attention).unsqueeze(-1)

        return torch.sum(feats * attention, dim=1)

    def forward(self, x):
        if self.layout == "legacy_view":
            return self._forward_legacy_view(x)
        return self._forward_stack(x)

class SKUnit(nn.Module):
    def __init__(
        self,
        in_features,
        mid_features,
        out_features,
        dim1,
        dim2,
        pool_dim,
        M=4,
        G=1,
        r=4,
        stride=1,
        L=32,
        use_min_l=False,
        layout="legacy_view",
    ):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_sk = nn.Sequential(
            SKConv(
                input_dim=mid_features,
                output_dim=out_features,
                dim1=dim1,
                dim2=dim2,
                pool_dim=pool_dim,
                M=M,
                G=G,
                r=r,
                stride=stride,
                L=L,
                use_min_l=use_min_l,
                layout=layout,
            ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            )
        
        
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, mid_features*2, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features*2),
            nn.ReLU(inplace=True),
            
            )
     

        if in_features == out_features: # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features,out_features , 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        #out = self.conv3(out)
        # print(f"Output shape: {out.size()}")
        # out = self.attention(out)
        #return self.relu(out + self.shortcut(residual))
        #return self.relu(out)
        return out
