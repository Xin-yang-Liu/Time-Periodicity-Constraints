import numpy as np
import torch 


class FD_derivative():
    def __init__(self, seq_len, dt, dx, meshx, meshy,BC,IC):
        self.seq_len = seq_len
        self.self.dt = dt
        self.dx = dx
        self.dx2 = dx*dx
        self.meshx = meshx
        self.meshy = meshy
        self.zero = torch.zeros(1, meshx, meshy)


    def dudt_f(self, u):
        dudt_first = u[0:1]-u[0]
        dudt_inner = (-0.5*u[0:-2]+0.5*u[2:])/self.dt
        dudt_last = (1.5*u[-1:]-2*u[-2:-1]+0.5*u[-3:-2])/self.dt
        dudt = torch.cat((dudt_first, dudt_inner, dudt_last))
        return dudt


    def dudx_upwind(self, u):
        dudx_left1 = 0*u[1:, :, 0:1]
        dudx_left2 = (-u[1:, :, 0:1] + u[1:, :, 1:2])/self.dx
        dudx_right = (1.5*u[1:, :, 2:]-2*u[1:, :, 1:-1]+0.5*u[1:, :, :-2])/self.dx
        dudx = torch.cat((dudx_left1, dudx_left2, dudx_right), 2)
        dudx = torch.cat((self.zero, dudx))
        return dudx


    def dudy_f(self, u):
        dudy_left = (-1.5*u[1:, 0:1, :]+2*u[1:, 1:2, :]-0.5*u[1:, 2:3, :])/self.dx
        dudy_inner = (-0.5*u[1:, :-2, :]+0.5*u[1:, 2:, :])/self.dx
        dudy_right = (1.5*u[1:, -1:, :]-2*u[1:, -2:-1, :]+0.5*u[1:, -3:-2, :])/self.dx
        dudy = torch.cat((dudy_left, dudy_inner, dudy_right), 1)
        dudy = torch.cat((self.zero, dudy))
        return dudy


    def d2udy2_f(self, u):
        d2udy2_left = (2*u[1:, 0:1, :]-5*u[1:, 1:2, :] +
                       4*u[1:, 2:3, :]-u[1:, 3:4, :])/self.dx2
        d2udy2_inner = (u[1:, :-2, :]-2*u[1:, 1:-1, :]+u[1:, 2:, :])/self.dx2
        d2udy2_right = (2*u[1:, -1:, :]-5*u[1:, -2:-1, :] +
                        4*u[1:, -3:-2, :]-u[1:, -4:-3, :])/self.dx2
        d2udy2 = torch.cat((d2udy2_left, d2udy2_inner, d2udy2_right), 1)
        d2udy2 = torch.cat((self.zero, d2udy2))
        return d2udy2


    def dudx_f(self, u):
        dudx_up = (-1.5*u[1:, :, 0:1]+2*u[1:, :, 1:2]-0.5*u[1:, :, 2:3])/self.dx
        dudx_inner = (-0.5*u[1:, :, :-2]+0.5*u[1:, :, 2:])/self.dx
        dudx_down = (1.5*u[1:, :, -1:]-2*u[1:, :, -2:-1]+0.5*u[1:, :, -3:-2])/self.dx
        dudx = torch.cat((dudx_up, dudx_inner, dudx_down), 2)
        dudx = torch.cat((self.zero, dudx))
        return dudx


    def d2udx2_f(self, u):
        d2udx2_up = (2*u[1:, :, 0:1]-5*u[1:, :, 1:2] +
                     4*u[1:, :, 2:3]-u[1:, :, 3:4])/self.dx2
        d2udx2_inner = (u[1:, :, :-2]-2*u[1:, :, 1:-1]+u[1:, :, 2:])/self.dx2
        d2udx2_down = (2*u[1:, :, -1:]-5*u[1:, :, -2:-1] +
                       4*u[1:, :, -3:-2]-u[1:, :, -4:-3])/self.dx2
        d2udx2 = torch.cat((d2udx2_up, d2udx2_inner, d2udx2_down), 2)
        d2udx2 = torch.cat((self.zero, d2udx2))
        return d2udx2

    # def zLD(self, x, num_BC):
    #     '''
    #     Overwrite Loss on Dirichlet Boundarys to Zeros
    #     '''
    #     for i in num_BC:
    #         if i == 0:
    #             x[:, :, 0] = 0*x[:, :, 0]
    #         elif i == 1:
    #             x[:, :, -1] = 0*x[:, :, -1]
    #         elif i == 2:
    #             x[:, 0, :] = 0*x[:, 0, :]
    #         elif i == 3:
    #             x[:, -1, :] = 0*x[:, -1, :]
    #     return x


    def forceBCICu(self, data, IC, BC_inout):
        #u = torch.cat((IC[1:-1,1:].view(1,meshx-2,meshy-1),data[1:,1:-1,1:]))
        y0 = torch.zeros(self.seq_len,self.meshy)
        u = data[:, 1:-1, 1:]
        u = torch.cat((BC_inout[:, 1:-1].view(self.seq_len, self.meshx-2, -1), u), 2)
        u = torch.cat((y0.view(self.seq_len, -1, self.meshy), u,
                        y0.view(self.seq_len, -1, self.meshy)), 1)
        return u



    def forceBCICp(self, data, IC, BC_inout):
        #u = torch.cat((IC[1:-1,:-1].view(1,meshx-2,meshy-1),data[1:,1:-1,:-1]))
        u = data[:, 1:-1, :-1]
        u = torch.cat((u[:, 0:1, :], u, u[:, -1:, :]), 1)
        u = torch.cat((u, BC_inout.view(seq_len, -1, 1)), 2)
        return u


    

if __name__=="__main__":
    a=FD_derivative
