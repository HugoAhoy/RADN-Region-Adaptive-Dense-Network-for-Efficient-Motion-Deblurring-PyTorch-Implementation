# Simple Idea
# offset_conv(out_channel = kernel_elnum*2)
# use grid_sample(a function in PyTorch) to transform feature map
# conv(offset_feature_map)
import torch
import torch.nn as nn
import numpy as np

class DeformableConv2d(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding = None):
        super(DeformableConv2d, self).__init__()
        self.inc = inchannel
        self.outc = outchannel
        self.kernel_size = kernel_size
        assert isinstance(kernel_size, int)

        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)

        self.kernel_elnum = kernel_size * kernel_size # here kernel_elnum refers to the N = |R| in DCN paper


        self.offset_conv = nn.Conv2d(inchannel, self.kernel_elnum * 2, kernel_size, padding)
        self.dconv = nn.Conv2d(inchannel, outchannel, kernel_size, kernel_size)
        self.init_weight()

    def forward(self, x):
        offset = self.offset_conv(x)
        expanded_feature_map = self.get_expanded(offset, x)
        out = self.dconv(expanded_feature_map)
        return out

    def get_expanded(self, offset, x):
        # this function is based on https://github.com/ChunhuanLin/deform_conv_pytorch
        assert offset.size(1) == self.kernel_elnum*2 #[B,C,H,W] --> C = N*2 = kernel_elnum*2
        dtype = offset.data.type()
        ks = self.kernel_size
        # N=kernel_elnum
        N = offset.size(1) // 2

        # 将offset的顺序从[x1, y1, x2, y2, ...] 改成[x1, x2, .... y1, y2, ...]
        offsets_index = torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]).type_as(x).long()

        # torch.unsqueeze()是为了增加维度
        # 经过unsqueeze，offsets_index维度变为[1,2N,1,1]
        # offset_size = [batch, 2N, featuremapH, featuremapW]
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # 根据维度dim按照索引列表index从offset中选取指定元素
        # ouput[b][c][h][w] = input[b][index[b][c][h][w]][h][w]
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        # 对输入x进行padding
        # TODO:where to padding? the conv or here or both
        if self.padding:
            x = self.zero_padding(x)

        # 将offset放到网格上，也即是图中所示的那个绿色带箭头的offset图
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # 维度变换
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # floor是向下取整
        q_lt = p.detach().floor()
        # +1相当于向上取整
        q_rb = q_lt + 1

        # 将lt限制在图像范围内
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # 将rb限制在图像范围内
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # 获得lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        # 获得rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # 插值的时候需要考虑一下padding对原始索引的影响
        # (b, h, w, N)
        # torch.lt() 逐元素比较input和other，即是否input < other
        # torch.rt() 逐元素比较input和other，即是否input > other
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        # 相当于求双线性插值中的u和v
        floor_p = p - (p - torch.floor(p))
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # 插值的4个系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 插值的最终操作在这里
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        return x_offset

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        # p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)
        p_n = torch.from_numpy(p_n).requires_grad_(False).type(dtype)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        # p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)
        p_0 = torch.from_numpy(p_0).requires_grad_(False).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        _, N, h, w = offset.size()
        N = N // 2

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # here use Tensor Broadcasting
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

    def init_weight(self):
        self.offset_conv.weight.zero_()
        if self.offset_conv.bias is not None:
            self.offset_conv.bias.zero_()
        self.dconv.weight.data = torch.ones(self.dconv.weight.size())