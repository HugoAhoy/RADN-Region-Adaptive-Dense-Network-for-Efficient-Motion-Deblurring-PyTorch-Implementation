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


class DensePart(nn.Module):
    def __init__(self):
        super(DensePart, self).__init__(num_input_features, growthrate)
        self.growthrate = growthrate
        self.bn_size = 4
        self.DenseBlock1 = DenseBlock(12, self.growthrate, num_input_features, self.bn_size)
        self.DenseBlock2 = DenseBlock(16, self.growthrate, self.growthrate //2, self.bn_size)
        self.DenseBlock3 = DenseBlock(24, self.growthrate, self.growthrate //2, self.bn_size)
        self.Transition1 = Transition(self.growthrate, self.growthrate // 2)
        self.Transition2 = Transition(self.growthrate, self.growthrate // 2)

    def forward(self, input):
        # TODO:x1 = space2depth(input) and change the "input" next line to x1
        x2 =  self.DenseBlock1(input) # correct is DenseBlock1(x1)
        x3 = self.Transition1(x2)
        x4 = self.DenseBlock2(x3)
        x5 = self.Transition2(x4)
        x6 = self.DenseBlock3(x5)
        return x2, x4, x6

class DDM(nn.Module):
    def __init__(self):
        super(DDM, self).__init__()
    def forward(self, input):
        pass

class RADN(nn.Module):
    def __init__(self):
        super(RADN, self).__init__()
        self.scale = 2
        self.inputconv = nn.Conv2d(12,12,(1,1),padding = 0)
        self.growthrate = 32
        self.DensePart = DensePart(3*4, self.growthrate)
        self.SA = SA()
        self.DDM1 = DDM(6, self.growthrate, self.growthrate // 2, 0.5)
        self.DDM2 = DDM(6, self.growthrate, self.growthrate // 2, 0.5)
        self.DDM3 = DDM(6, self.growthrate, self.growthrate // 2, 0.5)
        self.transconv1 = nn.ConvTranspose2d(self.growthrate // 2, self.growthrate // 2, kernel_size=2, stride=2)
        self.transconv2 = nn.ConvTranspose2d(self.growthrate // 2, self.growthrate // 2, kernel_size=2, stride=2)
        self.transconv3 = nn.ConvTranspose2d(self.growthrate // 2, self.growthrate // 2, kernel_size=2, stride=2)
        # TODO:self.reconstruction

    def forward(self, input):
        x = space_to_depth(input, self.scale)
        # process by an unknown conv layer
        x = self.inputconv(x)

        x1, x2, x3 = self.DensePart(x)
        x4 = self.SA(x3)
        x5 = self.transconv1(self.DDM1(x4))
        x6 = self.transconv2(self.DDM2(torch.cat([x2,x5],1)))
        x7 = self.transconv3(self.DDM3(torch.cat([x1,x6],1)))
        # TODO:reconstruction


if __name__ == "__main__":
    a = torch.arange(1 * 6 * 6 * 2).type(torch.float32).reshape(1, 2, 6, 6)
    scale = 2
    print(a)
    print(space_to_depth(a))