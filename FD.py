import numpy as np
import torch 

mytype = torch.float
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    mytype = torch.float

class para_data():
    '''
    2D pipe flow data
    ===
    input(seq_len: int, batch_size: int, dt: int, dx: int,
         N_meshx: int, N_meshy: int, initial: [meshx, meshy])
    '''

    def __init__(self, paramDict):
        for i in paramDict:
            try:
                paramDict[i].isdight()
            except:
                setattr(self, i, paramDict[i])
        initial = np.load(paramDict['initial'])
        self.initial = torch.tensor(initial,dtype=mytype).repeat([self.batch_size,1,1,1])
        self.y0 = torch.zeros(self.batch_size,self.seq_len, 1, self.meshy)
        self.dx2 = self.dx*self.dx

def dudt_f(u, pd):
    dudt_first = u[:,0:1] - pd.initial
    dudt_inner = (-0.5*u[:,0:-2]+0.5*u[:,2:])/pd.dt
    dudt_last = (1.5*u[:,-1:]-2*u[:,-2:-1]+0.5*u[:,-3:-2])/pd.dt
    dudt = torch.cat((dudt_first,dudt_inner,dudt_last),1)
    return dudt

def dudy_f(u, pd):
    dudy_left = (-1.5*u[:,:,0:1,:]+2*u[:,:,1:2,:]-0.5*u[:,:,2:3,:])/pd.dx
    dudy_inner = (-0.5*u[:,:,:-2,:]+0.5*u[:,:,2:,:])/pd.dx
    dudy_right = (1.5*u[:,:,-1:,:]-2*u[:,:,-2:-1,:]+0.5*u[:,:,-3:-2,:])/pd.dx
    dudy = torch.cat((dudy_left,dudy_inner,dudy_right),2)
    return dudy

def d2udy2_f(u, pd):
    d2udy2_left = (2*u[:,:,0:1,:]-5*u[:,:,1:2,:]+4*u[:,:,2:3,:]-u[:,:,3:4,:])/pd.dx2
    d2udy2_inner = (u[:,:,:-2,:]-2*u[:,:,1:-1,:]+u[:,:,2:,:])/pd.dx2
    d2udy2_right = (2*u[:,:,-1:,:]-5*u[:,:,-2:-1,:]+4*u[:,:,-3:-2,:]-u[:,:,-4:-3,:])/pd.dx2
    d2udy2 = torch.cat((d2udy2_left,d2udy2_inner,d2udy2_right),2)
    return d2udy2


def dudx_f(u, pd):
    dudx_up = (-1.5*u[:,:,:,0:1]+2*u[:,:,:,1:2]-0.5*u[:,:,:,2:3])/pd.dx
    dudx_inner = (-0.5*u[:,:,:,:-2]+0.5*u[:,:,:,2:])/pd.dx
    dudx_down = (1.5*u[:,:,:,-1:]-2*u[:,:,:,-2:-1]+0.5*u[:,:,:,-3:-2])/pd.dx
    dudx = torch.cat((dudx_up,dudx_inner,dudx_down),3)
    return dudx

def d2udx2_f(u, pd):
    d2udx2_up = (2*u[:,:,:,0:1]-5*u[:,:,:,1:2]+4*u[:,:,:,2:3]-u[:,:,:,3:4])/pd.dx2
    d2udx2_inner = (u[:,:,:,:-2]-2*u[:,:,:,1:-1]+u[:,:,:,2:])/pd.dx2
    d2udx2_down = (2*u[:,:,:,-1:]-5*u[:,:,:,-2:-1]+4*u[:,:,:,-3:-2]-u[:,:,:,-4:-3])/pd.dx2
    d2udx2 = torch.cat((d2udx2_up,d2udx2_inner,d2udx2_down),3)
    return d2udx2

def forceBCICu(data, BC_inout, pd):
    u = data[:,:,1:-1,1:]
    u = torch.cat((BC_inout[:,:,:, 1:-1].view(pd.batch_size,pd.seq_len, pd.meshx-2, -1), u), 3)
    u = torch.cat((pd.y0, u, pd.y0), 2)
    return u

def forceBCICp(data, BC_inout, pd):
    u = data[:,:,1:-1,:-1]
    u = torch.cat((u[:,:,0:1,:], u, u[:,:,-1:,:]),2)
    u = torch.cat((u,BC_inout.view(pd.batch_size,pd.seq_len,-1,1)),3)
    return u
        

if __name__=="__main__":
    pass
