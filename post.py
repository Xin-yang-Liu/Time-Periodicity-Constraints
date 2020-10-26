#Compare results from nn to openfoam 

from model import read2D
import matplotlib.pyplot as plt 
import torch
import numpy as np
import os
from utility import *


def plot(data, name, min=0, max=1,size=(10,4)):
    '''
    size: tuple
    '''
    plt.figure(figsize=size)
    plt.axis('equal')
    plt.pcolormesh(data,vmin=min,vmax=max)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(name + '.png',dpi=300,)
    plt.close()

def timeseriesdraw(name, dirs, ofre=None, num_fig=20, interval=10, convert=False):

    data = torch.load(name+'.pt', map_location=torch.device('cpu'))
    data = data.detach().numpy()
    data = np.array([data[(i+1)*interval-1] for i in range(num_fig)])
    if convert==True:
        data = (data[:,0:-1,:-1]+data[:,0:-1,1:]+data[:,1:,:-1]+data[:,1:,1:])/4
    
    chmkdir([dirs,name])
    if ofre.any()!=None: 
        error=data - ofre
        err_max,err_min=np.max(error),np.min(error)
        rel_error = np.abs(error)/ofre
        rel_err_max=np.max(rel_error)
    for t in range(data.shape[0]):
        if ofre.any()!=None: 
            plot(error[t], 'err_' + str(t),min=err_min,max=err_max)
            plot(rel_error[t], 'rel_err_' + str(t),max=rel_err_max)
            plot(ofre[t], 'opfm_' + str(t),min=0,max=1.2)
        plot(data[t], 'nn_' + str(t),min=0,max=1.2)


if __name__=='__main__':
    of=read2D('/home/lxy/test/openfoam/pipeflow')[:,:,:,0]
    timeseriesdraw('/home/lxy/Archive/dynamic/pipe/uconti100','/home/lxy/Archive/dynamic/pipe',ofre=of, convert=True)

