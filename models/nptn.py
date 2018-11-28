import torch.nn as nn 
import torch 
import torch.nn.functional as F

class NPTN_vanquish(nn.Module):
    expansion = 1

    def __init__(self,  in_ch, out_ch, G, k, pad, stride ):
        super(NPTN_vanquish, self ).__init__()
        self.conv1 = nn.Conv2d(in_ch, G*in_ch*out_ch, kernel_size=k, padding=pad, groups=in_ch, stride=stride, bias=True)
        self.transpool = nn.MaxPool3d((G, 1, 1))
        self.chanpool = nn.AvgPool3d((in_ch, 1, 1))
        # self.chanpool = nn.MaxPool3d((in_ch, 1, 1),stride=(1,1,1), dilation=(out_ch,1,1))
        # self.chanpool = nn.AvgPool3d((1, 1, 1))
        # self.bn1 = nn.BatchNorm2d(G*in_ch*out_ch)
        # self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=pool, stride=pool, groups=out_ch)
        # self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=pool, stride=pool, groups=out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # self.m1 = nn.PReLU()
        self.m2 = nn.PReLU()
        self.out_ch = out_ch
        self.G = G
        self.index = torch.LongTensor(in_ch*out_ch).cuda()

        index = 0
        for ii in range(in_ch):
                for jj in range(out_ch):
                        self.index[ii + jj*in_ch] = index
                        index+=1
        # print(self.index)



    def forward(self, x):

        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.m1(out)
        # out = self.drop(out)
        # print(out.data.size())

        out = self.transpool(out)
        out = out[:,self.index,:,:]
        # print(out.data.size())
        out = self.chanpool(out)


        # print(out.data.size())
        # out = self.conv2(out) # in_ch*out_ch*expansion  ->  out_ch
        # out = self.m2(out)
        # out = self.conv3(out) # out_ch  ->  out_ch
        # out = torch.sum(out, 1, keepdim=True)
        # out = self.conv2(out) # diff from db9 ori


        # print(out.data.size())
        # out += residual
        return out


# self.nptn1 = NPTN_vanquish(in_ch=3, out_ch=self.channels, G=self.G, k=5, pad=0, stride=1)
