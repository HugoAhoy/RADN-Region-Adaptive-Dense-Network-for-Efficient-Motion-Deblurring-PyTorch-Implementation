import torch
import torch.nn as nn
from network.deformmodule import DDM
from network.densemodule import DenseBlock, Transition
import torch.nn.functional as F
from network.samodule import SA

def space_to_depth(input, scale = 2):
    assert isinstance(input, torch.Tensor)
    assert isinstance(scale, int)
    b,c,h,w = input.size()
    assert h%scale == 0 and w%scale == 0
    unfolded_x = F.unfold(input, scale, stride=scale)
    return unfolded_x.view(b, c * scale ** 2, h // scale, w // scale)

def cat(x1, x2):
    # input is CHW
    assert x2.size()[2] >= x1.size()[2] and x2.size()[3] >= x1.size()[3]
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return x

class DensePart(nn.Module):
    def __init__(self,num_input_features, growthrate):
        super(DensePart, self).__init__()
        self.growthrate = growthrate
        self.bn_size = 4
        # assum input feature = 3*4 = 12, growthrate = 32
        # input = 12, output = 12*32+12 = 396
        self.DenseBlock1 = DenseBlock(12, self.growthrate, num_input_features, self.bn_size)
        self.out_feature_num1 = num_input_features + 12*growthrate
        self.Transition1 = Transition(self.out_feature_num1, self.out_feature_num1 // 2)
        # input = 198, output = 16*32+198 = 710
        self.DenseBlock2 = DenseBlock(16, self.growthrate, self.out_feature_num1 // 2, self.bn_size)
        self.out_feature_num2 = self.out_feature_num1//2 + 16*growthrate
        self.Transition2 = Transition(self.out_feature_num2, self.out_feature_num2 // 2)
        #input = 355, output = 24*32+355 = 1123
        self.DenseBlock3 = DenseBlock(24, self.growthrate, self.out_feature_num2 //2, self.bn_size)

    def forward(self, input):
        x2 =  self.DenseBlock1(input)
        # print(x2.size())
        x3 = self.Transition1(x2)
        # print(x3.size())
        x4 = self.DenseBlock2(x3)
        # print(x4.size())
        x5 = self.Transition2(x4)
        # print(x5.size())
        x6 = self.DenseBlock3(x5)
        # print(x6.size())
        # 396, 710, 1123, respectively
        return x2, x4, x6

class RADN(nn.Module):
    def __init__(self):
        super(RADN, self).__init__()
        self.scale = 2
        self.inputconv = nn.Conv2d(12,12,(1,1),padding = 0)
        self.growthrate = 32
        self.DensePart = DensePart(3*4, self.growthrate)
        # assume growthrate = 32
        # input(x4) = 1123, output = (6*32+input)*0.5 = 657
        self.SA = SA(1123)
        self.DDM1 = DDM(6, self.growthrate, 1123, 0.5)
        # input = 657, output = 128
        self.transconv1 = nn.ConvTranspose2d(657, 128, kernel_size=2, stride=2)
        # input(x2,x5) = 710+128, output = (6*32+input)*0.5 = 515
        self.DDM2 = DDM(6, self.growthrate, 838, 0.5)
        # input = 515, output = 128
        self.transconv2 = nn.ConvTranspose2d(515, 128, kernel_size=2, stride=2)
        # input(x1,x6) = 396+128, output = (6*32+input)*0.5 = 358
        self.DDM3 = DDM(6, self.growthrate, 524, 0.5)
        # input = 358, output = 64
        # because the output of last DDM will be concatenated with input, so the #channels is 64-3
        self.transconv3 = nn.ConvTranspose2d(358, 61, kernel_size=2, stride=2)

        # for multi-scale context aggregation(pooling and upsampling)
        self.pooling1_4 = torch.nn.AvgPool2d((4,4))
        self.pooling1_8 = torch.nn.AvgPool2d((8,8))
        self.pooling1_16 = torch.nn.AvgPool2d((16,16))
        self.pooling1_32 = torch.nn.AvgPool2d((32,32))

        self.upsampling32_16 = nn.ConvTranspose2d(self.growthrate*2, self.growthrate*2, kernel_size=2, stride=2)
        self.upsampling16_8 = nn.ConvTranspose2d(self.growthrate*4, self.growthrate*2, kernel_size=2, stride=2)
        self.upsampling8_4 = nn.ConvTranspose2d(self.growthrate*4, self.growthrate*2, kernel_size=2, stride=2)
        self.upsampling4_2 = nn.ConvTranspose2d(self.growthrate*4, self.growthrate*2, kernel_size=2, stride=2)
        # TODO:self.reconstruction
        self.upsampling2_1 = nn.ConvTranspose2d(self.growthrate*2, 3, kernel_size=2, stride=2)

    def forward(self, input):
        x = space_to_depth(input, self.scale)
        # process by an unknown conv layer
        x = self.inputconv(x)

        x1, x2, x3 = self.DensePart(x)
        x4 = self.SA(x3)
        x5 = self.transconv1(self.DDM1(x4))
        x6 = self.transconv2(self.DDM2(cat(x5,x2)))
        x7 = self.transconv3(self.DDM3(cat(x6,x1)))
        x8 = cat(x7,input)
        # multi-scale context aggregation? Not sure if it's correct
        p1_4 = self.pooling1_4(x8)
        p1_8 = self.pooling1_8(x8)
        p1_16 = self.pooling1_16(x8)
        p1_32 = self.pooling1_32(x8)

        x9 = self.upsampling32_16(p1_32)
        x10 = self.upsampling16_8(cat(x9, p1_16))
        x11 = self.upsampling8_4(cat(x10, p1_8))
        x12 = self.upsampling4_2(cat(x11, p1_4))
        
        # reconstruction
        # res is the residual between sharp and blur
        res = self.upsampling2_1(x12)
        diffY = input.size()[2] - res.size()[2]
        diffX = input.size()[3] - res.size()[3]

        res = F.pad(res, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out = res+input
        return out

if __name__ == "__main__":
    a = torch.arange(1 * 6 * 6 * 2).type(torch.float32).reshape(1, 2, 6, 6)
    scale = 2
    print(a)
    print(space_to_depth(a))