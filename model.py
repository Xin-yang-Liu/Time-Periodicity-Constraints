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
from torch.utils.data import Dataset, DataLoader
import math


class cnn2(nn.Module):
    def __init__(self, data):
        super(cnn2, self).__init__()

        self.l = nn.Linear(1, data.meshy)
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 15, 3, padding=1)
        self.c3 = nn.Conv2d(15, 9, 3, padding=1)
        self.c4 = nn.Conv2d(9, 3, 3, padding=1)
        self._initialize_weights()
        self.relu = nn.ReLU()


    def forward(self, x):
        '''
        input shape [seq_len,channel,meshx,1]
        '''
        x = self.l(x)
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
        x = self.relu(self.c3(x))
        x = self.c4(x)
        
        return x[:,0,:,:],x[:,1,:,:],x[:,2,:,:]

    def _initialize_weights(self):
        init.kaiming_normal_(self.c1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.c2.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.c3.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.c4.weight)


class NSLSTM(nn.Module):
    '''
        self.in_size = in_size  
        self.out_size = out_size
    '''
    def __init__(self, in_size, out_size):
        super(NSLSTM, self).__init__()
        self.unet = nn.LSTM(in_size, out_size, 3)
        self.vnet = nn.LSTM(in_size, out_size, 3)
        self.pnet = nn.LSTM(in_size, out_size, 3)
        self.uh1 = nn.Linear(in_size, out_size)
        self.vh1 = nn.Linear(in_size, out_size)
        self.ph1 = nn.Linear(in_size, out_size)
        

    def forward(self, uin, vin, pin,uh,vh,ph):
        uh = self.uh1(uh)
        vh = self.vh1(vh)
        ph = self.ph1(ph)
        u,_=self.unet(uin,(uh,uh))
        v,_=self.vnet(vin,(vh,vh))
        p,_=self.pnet(pin,(ph,ph))
        return u,v,p

class CNNLSTM(nn.Module):
    def __init__(self, pd):
        super(CNNLSTM, self).__init__()
        self.batch_size = pd.batch_size
        self.seq_len = pd.seq_len
        self.meshx = pd.meshx
        self.meshy = pd.meshy
        self.ul = nn.Linear(1, self.meshy)
        self.vl = nn.Linear(1, self.meshy)
        self.pl = nn.Linear(1, self.meshy)
        self.c1 = nn.Conv3d(3, 8,kernel_size=(1,3,3),padding=(0,1,1))
        self.c2 = nn.Conv3d(8,15,kernel_size=(1,3,3),padding=(0,1,1))
        self.c3 = nn.Conv3d(15,8,kernel_size=(1,3,3),padding=(0,1,1))
        self.c4 = nn.Conv3d(8, 3,kernel_size=(1,3,3),padding=(0,1,1))
        self.ulstm = nn.LSTM(self.meshx*self.meshy,self.meshx*self.meshy,1,batch_first=True)
        self.vlstm = nn.LSTM(self.meshx*self.meshy,self.meshx*self.meshy,1,batch_first=True)
        self.plstm = nn.LSTM(self.meshx*self.meshy,self.meshx*self.meshy,1,batch_first=True)
        self.relu = nn.ReLU()
        self._initialize_weights()
        self.pool = nn.MaxPool2d(3,stride=1)
        self.ux = nn.Parameter(torch.randn(1,1,self.meshx*self.meshy))
        self.vx = nn.Parameter(torch.randn(1,1,self.meshx*self.meshy))
        self.px = nn.Parameter(torch.randn(1,1,self.meshx*self.meshy))
        

    def forward(self,ubc,vbc,pbc):
        '''
        BC:[seq_len,meshx,1]
        '''
        u = self.ul(ubc)
        v = self.vl(vbc)
        p = self.pl(pbc)
        x = torch.cat((u,v,p),1)
        
        x = self.relu(self.c1(x))

        x = self.relu(self.c2(x))
        x = self.relu(self.c3(x))
        x = self.c4(x)
        u,v,p = x[:,0].view(self.batch_size,self.seq_len,-1),x[:,1].view(self.batch_size,self.seq_len,-1),x[:,2].view(self.batch_size,self.seq_len,-1)
        
        uic =self.ux.repeat([1,self.batch_size,1])
        vic =self.vx.repeat([1,self.batch_size,1])
        pic =self.px.repeat([1,self.batch_size,1])
        
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
    
    net = CNNLSTM(3)
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    loss_f = nn.MSELoss()
    data = mydata()
    dataloader = DataLoader(dataset=data, batch_size=3, shuffle=True, num_workers=3)
    for i in range(10):
        for inp in dataloader:
            ubc,vbc,pbc = inp
            u,v,p,uic = net(ubc,vbc,pbc)
            print(uic[0,0,0])
            u = u.view(3,100,11,51)
            l1=loss_f(u,torch.zeros(3,100,11,51))
            l1.backward()
            optimizer.step()
    torch.save(net.state_dict(),'test')
    net2 = CNNLSTM(1)
    net2.load_state_dict(torch.load('test'))
    print()



