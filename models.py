import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Up_Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_Conv_block, self).__init__()
        self.upConv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.upConv(x)

"""
    U-Net
    https://arxiv.org/pdf/1505.04597.pdf
"""
class U_Net(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super(U_Net, self).__init__()

        output_channels = [64, 128, 256, 512, 1024]
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv_block(in_channels=input_channels, out_channels=output_channels[0])
        self.Conv2 = Conv_block(in_channels=output_channels[0], out_channels=output_channels[1])
        self.Conv3 = Conv_block(in_channels=output_channels[1], out_channels=output_channels[2])
        self.Conv4 = Conv_block(in_channels=output_channels[2], out_channels=output_channels[3])
        self.Conv5 = Conv_block(in_channels=output_channels[3], out_channels=output_channels[4])

        self.Up5 = Up_Conv_block(in_channels=output_channels[4], out_channels=output_channels[3])
        self.Up_Conv5 = Conv_block(in_channels=output_channels[4], out_channels=output_channels[3])

        self.Up4 = Up_Conv_block(in_channels=output_channels[3], out_channels=output_channels[2])
        self.Up_Conv4 = Conv_block(in_channels=output_channels[3], out_channels=output_channels[2])

        self.Up3 = Up_Conv_block(in_channels=output_channels[2], out_channels=output_channels[1])
        self.Up_Conv3 = Conv_block(in_channels=output_channels[2], out_channels=output_channels[1])

        self.Up2 = Up_Conv_block(in_channels=output_channels[1], out_channels=output_channels[0])
        self.Up_Conv2 = Conv_block(in_channels=output_channels[1], out_channels=output_channels[0])

        self.Conv_1x1 = nn.Conv2d(in_channels=output_channels[0], out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Encoding
        x1 = self.Conv1(x)

        x2 = self.MaxPool(x1)
        x2 = self.Conv2(x2)

        x3 = self.MaxPool(x2)
        x3 = self.Conv3(x3)

        x4 = self.MaxPool(x3)
        x4 = self.Conv4(x4)

        x5 = self.MaxPool(x4)
        x5 = self.Conv5(x5)

        # Decoding and Concating
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_Conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_Conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_Conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_Conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


"""
    U-Net++
    https://arxiv.org/pdf/1807.10165.pdf
"""
class Nested_UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super(Nested_UNet, self).__init__()

        output_channels = [64, 128, 256, 512, 1024]
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv0_0 = Conv_block(in_channels=input_channels, out_channels=output_channels[0])
        self.Conv1_0 = Conv_block(in_channels=output_channels[0], out_channels=output_channels[1])
        self.Conv2_0 = Conv_block(in_channels=output_channels[1], out_channels=output_channels[2])
        self.Conv3_0 = Conv_block(in_channels=output_channels[2], out_channels=output_channels[3])
        self.Conv4_0 = Conv_block(in_channels=output_channels[3], out_channels=output_channels[4])

        self.Conv0_1 = Conv_block(output_channels[0]*1 + output_channels[1], output_channels[0])
        self.Conv0_2 = Conv_block(output_channels[0]*2 + output_channels[1], output_channels[0])
        self.Conv0_3 = Conv_block(output_channels[0]*3 + output_channels[1], output_channels[0])
        self.Conv0_4 = Conv_block(output_channels[0]*4 + output_channels[1], output_channels[0])

        self.Conv1_1 = Conv_block(output_channels[1]*1 + output_channels[2], output_channels[1])
        self.Conv1_2 = Conv_block(output_channels[1]*2 + output_channels[2], output_channels[1])
        self.Conv1_3 = Conv_block(output_channels[1]*3 + output_channels[2], output_channels[1])

        self.Conv2_1 = Conv_block(output_channels[2]*1 + output_channels[3], output_channels[2])
        self.Conv2_2 = Conv_block(output_channels[2]*2 + output_channels[3], output_channels[2])

        self.Conv3_1 = Conv_block(output_channels[3] + output_channels[4], output_channels[3])

        self.Up = nn.UpsamplingBilinear2d(scale_factor=2)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(in_channels=output_channels[0], out_channels=num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(in_channels=output_channels[0], out_channels=num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(in_channels=output_channels[0], out_channels=num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(in_channels=output_channels[0], out_channels=num_classes, kernel_size=1)
        else:
            self.Conv_1x1 = nn.Conv2d(in_channels=output_channels[0], out_channels=num_classes, kernel_size=1)


    def forward(self, Input):
        x0_0 = self.Conv0_0(Input)
        x0_0 = self.MaxPool(x0_0)
        x1_0 = self.Conv1_0(x0_0)
        x0_1 = self.Conv1_0(torch.cat((x0_0, self.Up(x1_0)), dim=1))

        x2_0 = self.Conv2_0(x1_0)
        x1_1 = self.Conv1_1(torch.cat((x1_0, self.Up(x2_0)), dim=1))
        x0_2 = self.Conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], dim=1))

        x3_0 = self.Conv3_0(x2_0)
        x2_1 = self.Conv2_1(torch.cat([x2_0, self.Up(x3_0)], dim=1))
        x1_2 = self.Conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], dim=1))
        x0_3 = self.Conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], dim=1))

        x4_0 = self.Conv4_0(x3_0)
        x3_1 = self.Conv3_1(torch.cat([x3_0, self.Up(x4_0)], dim=1))
        x2_2 = self.Conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], dim=1))
        x1_3 = self.Conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], dim=1))
        x0_4 = self.Conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output = [output1, output2, output3, output4]

        else:
            output = self.Conv_1x1(x0_4)

        return output



























