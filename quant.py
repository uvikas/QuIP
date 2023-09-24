import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import method

def quantize_qfna(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def quantize_qfnb(x, scale, maxq):
    q = x / scale
    q = torch.clamp(torch.round(((q+1)/2) * maxq), 0, maxq)
    q = (q / maxq) * 2 - 1
    q = q * scale
    return q

def quantize_qfnc(x, scale, zero, maxq):
    # for LDL vs GPTQ equivalency
    q = torch.clamp((x / scale) + zero, 0, maxq)
    q = torch.round(q)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.scaleWH = None

    def configure(self,
                  bits,
                  perchannel=False,
                  sym=True,
                  qfn='a',
                  mse=False,
                  norm=2.4,
                  grid=100,
                  maxshrink=.8):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.qfn = qfn
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        if self.qfn == 'a':
            self.find_params_qfna(x, weight=weight)
        elif self.qfn == 'b':
            self.find_params_qfnb(x)
        elif self.qfn == 'c':
            self.find_params_qfna(x, weight=weight)

    def find_params_qfna(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 /
                                    scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1),
                             self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def find_params_qfnb(self, x):
        dev = x.device
        self.maxq  = self.maxq.to(dev)
        self.scale = 2.4 * x.square().mean().sqrt() + 1e-16 
        self.zero  = None

    def quantize(self, x):
        if self.qfn == 'a':
            assert self.ready()
            return quantize_qfna(x, self.scale, self.zero, self.maxq)
        elif self.qfn == 'b':
            assert torch.all(self.maxq != 0)
            self.scale = 2.4 * x.square().mean().sqrt() + 1e-16
            return quantize_qfnb(x, self.scale, self.maxq)
        elif self.qfn == 'c':
            # for LDL vs GPTQ equivalency, does round in same order as bal code
            assert self.ready()
            return quantize_qfnc(x, self.scale, self.zero, self.maxq)
        else:
            return NotImplementedError()

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
except:
    print('CUDA extension not installed.')


# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module):

    def __init__(self, infeatures, outfeatures):
        super().__init__()
        # self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scaleWH', torch.zeros((infeatures)))
        self.register_buffer('scale', torch.zeros(()))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight',
            torch.zeros((infeatures, outfeatures),
                        dtype=torch.int))
        
        # self.register_buffer(
        #     'qweight',
        #     torch.zeros((infeatures // 32 * 3, outfeatures),
        #                 dtype=torch.int))

    def pack(self, linear, scale, scaleWH):
        # self.zeros = zeros * scales
        # self.scales = scales.clone()
        assert linear.bias.shape == self.bias.shape
        self.bias = linear.bias.clone()
        
        assert self.scale.shape == scale.shape
        self.scale = scale.reshape(1).clone()
        
        assert self.scaleWH.shape == scaleWH.shape
        self.scaleWH = scaleWH.clone()
        # intweight = torch.round(
        #     (linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = linear.weight.data.t().contiguous()
        
        # intweight = intweight.numpy().astype(np.uint32)
        # qweight = np.zeros(
        #     (intweight.shape[0] // 32 * 3, intweight.shape[1]),
        #     dtype=np.uint32)
        # i = 0
        # row = 0
        # while row < qweight.shape[0]:
        #     for j in range(i, i + 10):
        #         qweight[row] |= intweight[j] << (3 * (j - i))
        #     i += 10
        #     qweight[row] |= intweight[i] << 30
        #     row += 1
        #     qweight[row] |= (intweight[i] >> 2) & 1
        #     i += 1
        #     for j in range(i, i + 10):
        #         qweight[row] |= intweight[j] << (3 * (j - i) + 1)
        #     i += 10
        #     qweight[row] |= intweight[i] << 31
        #     row += 1
        #     qweight[row] |= (intweight[i] >> 1) & 0x3
        #     i += 1
        #     for j in range(i, i + 10):
        #         qweight[row] |= intweight[j] << (3 * (j - i) + 2)
        #     i += 10
        #     row += 1

        # qweight = qweight.astype(np.int32)
        # self.qweight = torch.from_numpy(qweight)
        self.qweight = intweight.to(torch.int)
    
    def dequantize(self):
        # Postprocessing
        t = time.perf_counter()
        w = self.qweight.t().to(torch.half)
        w = (w / (2**4-1)) * 2 - 1
        w = w * self.scale
        print('\tscale', time.perf_counter() - t)
        
        # Apply U and V to revert incoherence
        t = time.perf_counter()
        torch.manual_seed(0xCADE)
        torch.cuda.manual_seed(0xCADE)
        np.random.seed(0xCADE)
        print('\tseed', time.perf_counter() - t)
        
        t = time.perf_counter()
        U = method.rand_ortho_butterfly(w.shape[0]).to(torch.half).to(w.device)
        V = method.rand_ortho_butterfly(w.shape[1]).to(torch.half).to(w.device)
        w = (U.T @ w @ V)
        print('\tincoh', time.perf_counter() - t)
        
        # Revert diagonal scaling 
        t = time.perf_counter()
        w = (w / self.scaleWH[None,:]).to(torch.half)
        print('\tdiag scaling', time.perf_counter() - t)
        return w

    def forward(self, x):
        # print(x.shape)
        w = self.dequantize()
        return F.linear(x, w, self.bias)
        
        
        # if x.shape[-1] == x.numel():
        #     outshape = list(x.shape)
        #     y = self.bias.clone()
        #     outshape[-1] = self.bias.numel()
        #     dtype = x.dtype
        #     x = x.float()
        #     quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales,
        #                                self.zeros)
        #     y = y.to(dtype)
        #     return y.reshape(outshape)
        # raise ValueError('Only supports a single token currently.')


def make_quant3(module, names, name=''):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(module, attr,
                    Quant3Linear(tmp.in_features, tmp.out_features))
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1)
