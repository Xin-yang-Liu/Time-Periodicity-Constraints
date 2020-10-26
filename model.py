#python pkgs
import sys
import os

#3rd part pkgs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

#my pkgs
import utility


class cnn2(nn.Module):
    def __init__(self):
        super(cnn2, self).__init__()
        #self.l1 = nn.Conv2d(3, 8, 3, padding=2)
        #self.l2 = nn.Conv2d(8, 15, 3, padding=2)
        #self.l3 = nn.Conv2d(15, 9, 3, padding=2)
        self.l1 = nn.Conv2d(3, 8, 3, padding=1)
        self.l2 = nn.Conv2d(8, 15, 3, padding=1)
        self.l3 = nn.Conv2d(15, 9, 3, padding=1)
        self.l4 = nn.Conv2d(9, 3, 3, padding=1)
        self._initialize_weights()
        self.relu = nn.ReLU()
        #self.pool = nn.MaxPool2d(3,stride=1)
        #self.BC1 = 

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.l4(x)
        '''
        v = self.relu(self.l1(v))
        v = self.relu(self.l2(v))
        v = self.relu(self.l3(v))
        v = self.l4(v)
        p = self.relu(self.l1(p))
        p = self.relu(self.l2(p))
        p = self.relu(self.l3(p))
        p = self.l4(p)
        '''
        return x[:,0,:,:],x[:,1,:,:],x[:,2,:,:]

    def _initialize_weights(self):
        init.kaiming_normal_(self.l1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.l2.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.l3.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.l4.weight)

class NSLSTM(nn.Module):
    '''
    LSTM for 2D NS equation, combine 3 functions
    '''
    def __init__(self, in_size, out_size, layer):
        super(NSLSTM, self).__init__()
        '''
        self.in_size = in_size
        self.out_size = out_size
        '''
        self.unet = nn.LSTM(in_size, out_size, layer)
        self.vnet = nn.LSTM(in_size, out_size, layer)
        self.pnet = nn.LSTM(in_size, out_size, layer)

    def forward(self, uin, vin, pin, hn, cn):
        u,_=self.unet(uin,(hn,cn))
        v,_=self.vnet(vin,(hn,cn))
        p,_=self.pnet(pin,(hn,cn))
        return u,v,p

class CNNLSTM(nn.Module):
    def __init__(self,seq_len,meshx,meshy):
        super(CNNLSTM, self).__init__()
        self.seq_len = seq_len
        self.meshx = meshx
        self.meshy = meshy
        self.ul = nn.Linear(1, meshy)
        self.vl = nn.Linear(1, meshy)
        self.pl = nn.Linear(1, meshy)
        self.c1 = nn.Conv2d(3, 8,3,padding=1)
        self.c2 = nn.Conv2d(8,15,3,padding=1)
        self.c3 = nn.Conv2d(15,8,3,padding=1)
        self.c4 = nn.Conv2d(8, 3,3,padding=1)
        self.ulstm = nn.LSTM(meshx*meshy,meshx*meshy,1)
        self.vlstm = nn.LSTM(meshx*meshy,meshx*meshy,1)
        self.plstm = nn.LSTM(meshx*meshy,meshx*meshy,1)
        self.relu = nn.ReLU()
        self._initialize_weights()
        self.pool = nn.MaxPool2d(3,stride=1)
    def forward(self,ubc,vbc,pbc,uic,vic,pic):
        '''
        BC:[seq_len,meshx,1], IC:[1,1,meshx*meshy]
        '''
        u = self.ul(ubc).view(self.seq_len, 1, self.meshx, self.meshy)
        v = self.vl(vbc).view(self.seq_len, 1, self.meshx, self.meshy)
        p = self.pl(pbc).view(self.seq_len, 1, self.meshx, self.meshy)
        x = torch.cat((u,v,p),1)
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
        x = self.relu(self.c3(x))
        x = self.c4(x)
        u,v,p = x[:,0,:,:].view(self.seq_len,1,-1),x[:,1,:,:].view(self.seq_len,1,-1),x[:,2,:,:].view(self.seq_len,1,-1)
        u,_ = self.ulstm(u,(uic,uic))
        v,_ = self.vlstm(v,(vic,vic))
        p,_ = self.plstm(p,(pic,pic))
        return u,v,p

    def _initialize_weights(self):
        init.kaiming_normal_(self.ul.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.vl.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.pl.weight, mode='fan_out', nonlinearity='relu')

        init.kaiming_normal_(self.ulstm.weight_ih_l0)
        init.kaiming_normal_(self.ulstm.weight_hh_l0)
        init.kaiming_normal_(self.vlstm.weight_ih_l0)
        init.kaiming_normal_(self.vlstm.weight_hh_l0)
        init.kaiming_normal_(self.plstm.weight_ih_l0)
        init.kaiming_normal_(self.plstm.weight_hh_l0)
        
        init.kaiming_normal_(self.c1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.c2.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.c3.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.c4.weight)


def zLD(x,num_BC):
    '''
    Overwrite Loss on Dirichlet Boundarys to Zeros
    '''
    for i in num_BC:
        if i == 0:
            x[:,:,0] = 0*x[:,:,0]
        elif i == 1:
            x[:,:,-1] = 0*x[:,:,-1]
        elif i == 2:
            x[:,0,:] = 0*x[:,0,:]
        elif i == 3:
            x[:,-1,:] = 0*x[:,-1,:]
    return x

#######################################################################

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        #assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),#.cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))#.cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


if __name__=='__main__':
    net = ConvLSTM(3,[3,3],3,200,torch.linspace(0,199,200,dtype=int))
    inp = torch.ones((1,3,5,5))
    out=net(inp)
    loss = nn.MSELoss()



