import torch.nn as nn
import torch

class SA(nn.Module):
    def __init__(self, num_features):
        super(SA, self).__init__()
        self.num_features = num_features
        self.query_conv = nn.Conv2d(num_features, num_features//8, 1, bias = False)
        self.key_conv = nn.Conv2d(num_features, num_features//8, 1, bias = False)
        self.value_conv = nn.Conv2d(num_features, num_features//8, 1, bias = False)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, input):
        batchsize, C, H, W = input.size()
        query = self.query_conv(input).view(batchsize, -1, H*W).permute(0,2,1) # B*N*C/8
        key = self.key_conv(input).view(batchsize, -1, H*W) # B*C/8*N
        energy = torch.bmm(query, key)  # batch的matmul B*N*N
        attention = self.softmax(energy)
        value = self.value_conv(input).view(batchsize, -1, H*W)  # B * C * N
        out = torch.bmm(value,attention.permute(0,2,1)) # B*C*N
        out = out.view(batchsize,C,H,W) # B*C*H*W
        return out
