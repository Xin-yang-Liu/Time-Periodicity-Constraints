import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from FD import *
from model import *
from utility import *



torch.manual_seed(10)
mytype = torch.float
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    mytype = torch.float

inputfile = sys.argv[1]
paraDict=readInputData(inputfile)

pd = para_data(paraDict)


xvar = torch.linspace(0,math.pi,pd.meshx)
tvar = torch.linspace(6*math.pi/pd.seq_len,6*math.pi,pd.seq_len)
tgrid,xgrid = torch.meshgrid(tvar,xvar)

class mydata(Dataset):
    def __init__(self):
        self.u = torch.zeros(20,1,pd.seq_len,pd.meshx,1)
        self.v = torch.zeros(20,1,pd.seq_len,pd.meshx,1)
        self.p = torch.zeros(20,1,pd.seq_len,pd.meshx,1)
        self.len = 20
        for i in range(10):
            for j in range(5):
                self.u[i] = (0.5+0.5*torch.cos(0.4*j*tgrid)*torch.sin(xgrid+0.2*i*tgrid)).view(1,pd.seq_len,pd.meshx,1)
        
        
    def __getitem__(self, index):
        return self.u[index],self.v[index],self.p[index]

    def __len__(self):
        return self.len

net = CNNLSTM(pd)
optimizer = torch.optim.Adam(net.parameters(),lr=pd.learning_rate)
data = mydata()
dataloader = DataLoader(dataset=data, batch_size=pd.batch_size, shuffle=True, num_workers=0)
loss_f = nn.MSELoss()
zeros = torch.zeros([pd.batch_size,pd.seq_len, pd.meshx, pd.meshy])

start = time.time()

for i in range(10000):
    for inp in dataloader:
        optimizer.zero_grad()
        ubc,vbc,pbc = inp
        u,v,p = net(ubc,vbc,pbc)


        u = u.view(pd.batch_size,pd.seq_len, pd.meshx, pd.meshy)
        v = v.view(pd.batch_size,pd.seq_len, pd.meshx, pd.meshy)
        p = p.view(pd.batch_size,pd.seq_len, pd.meshx, pd.meshy)
    
        
        u=forceBCICu(u,ubc,pd)
        v=forceBCICu(v,vbc,pd)
        p=forceBCICp(p,pbc,pd)
    
        momentumx = dudt_f(u,pd) + u*dudx_f(u,pd) + v*dudy_f(u,pd) + dudx_f(p,pd) - 0.1*d2udx2_f(u,pd)-0.1*d2udy2_f(u,pd)
        momentumy = dudt_f(v,pd) + u*dudx_f(v,pd) + v*dudy_f(v,pd) + dudy_f(p,pd) - 0.1*d2udx2_f(v,pd)-0.1*d2udy2_f(v,pd)
        continuity = dudx_f(u,pd) + dudy_f(v,pd)
        # flowrate
    
        loss1 = loss_f(momentumx, zeros)
        loss2 = loss_f(momentumy, zeros)
        loss3 = loss_f(continuity,zeros)

        loss = loss1 + loss2 + pd.continuity_weight*loss3 
    
        loss.backward()
        optimizer.step()

  
    if i%100==0:
        print(loss.item())


# name = 'CNNlstmHardconstrained1'
# description = 'using 1 linear layer for hidden, 4 layer CNN, 1 layer lstm\n 11*51 mesh, temporal & spacial changing velocity inlet,\n Re=100, continuity weight 100, no upwind'
# try:
#   os.mkdir(name)
# except:
#   pass
# os.chdir(name)
# f=open('readme.txt','w')
# f.write(description)
# f.close()
print(time.time()-start)
torch.save(net.state_dict(),'CNN3lstm')