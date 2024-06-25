import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys
sys.path.insert(0, "./")  # run under the project directory
import numpy as np
from collections import OrderedDict

from common.network import *

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "c": 3, "t": 3, "e": 3, "l": 3, "f": 4, "m": 4, "g": 5, "n": 5}

def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable) with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

class SRNets(nn.Module):
    def __init__(self, opt, outC=1):
        super(SRNets, self).__init__()
        nf = opt.nf
        scale = 1
        self.modes = self.modes
        self.stages = opt.stages
        self.norm = opt.norm

        for s in range(self.stages): # 2-stage
            if (s+1) == self.stages:
                upscale = scale
                oC = outC
                flag = "N"
            else:
                upscale = None
                flag = "1"
                oc = 1
            for mode in self.modes:
                self.add_module("s{}_{}".format(str(s+1), mode), SRNet("{}x{}".format(mode.upper(), flag), nf=nf, upscale=upscale, outC=oC))
        # print_network(self)

    def forward(self, x, stage, mode):
        key = "s{}_{}".format(str(stage), mode)
        module = getattr(self, key)
        return module(x)

    def predict(self, x, stage=None):
        # Stage 1
        for s in range(self.stages):
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                for r in [0, 1, 2, 3]:
                    pred += round_func(torch.rot90(self.forward(F.pad(torch.rot90(x, r, [
                        2, 3]), (0, pad, 0, pad), mode='replicate'), stage=s+1, mode=mode), (4 - r) % 4, [2, 3]) * (self.norm//2))
            if (s+1 == self.stages):
                avg_factor, bias, norm = len(self.modes), 0, 1
            else:
                avg_factor, bias, norm = len(self.modes) * 4, self.norm//2, float(self.norm)
            x = torch.clamp(round_func((pred / avg_factor) + bias, 0, self.norm)) / norm

        return x


class SRNetsSWF2(nn.Module):
    def __init__(self, opt, inC=1, outC=1):
        super(SRNetsSWF2, self).__init__()
        nf = opt.nf
        scale = 1
        self.modes2 = opt.modes2
        self.modes = opt.modes
        self.stages = opt.stages
        self.norm = opt.norm

        for s in range(self.stages): # 2-stage
            if (s+1) == self.stages:
                upscale = scale
                oC = outC
                flag = "N"
                for mode in self.modes2:
                    for r in [0, 1]:
                        self.add_module("s{}_{}r{}".format(str(s+1), mode, r), SRNet("{}x{}".format(mode.upper(), flag), nf=nf, upscale=upscale, outC=oC))
            else:
                upscale = None
                flag = "1"
                oC = 1
                for mode in self.modes:
                    self.add_module("s{}_{}r0".format(str(s+1), mode), SRNet("{}x{}".format(mode.upper(), flag), nf=nf, upscale=upscale, outC=oC))
        # print_network(self)

    def forward(self, x, stage, mode, r):
        key = "s{}_{}r{}".format(str(stage), mode, r)
        module = getattr(self, key)
        return module(x)

    def predict(self, x, stage=None):
        if stage == 2: # hyper stage
            pred = 0
            for mode in self.modes2:
                pad = mode_pad_dict[mode]
                for r in [0, 2]:
                    pred += round_func(torch.rot90(self.forward(F.pad(torch.rot90(x, r, [
                        2, 3]), (0, pad, 0, pad), mode='replicate'), stage=self.stages, mode=mode, r=0), (4 - r) % 4, [2, 3]) * (self.norm//2))
                for r in [1, 3]:
                    pred += round_func(torch.rot90(self.forward(F.pad(torch.rot90(x, r, [
                        2, 3]), (0, pad, 0, pad), mode='replicate'), stage=self.stages, mode=mode, r=1), (4 - r) % 4, [2, 3]) * (self.norm//2))
            avg_factor, bias, norm = len(self.modes2) * 4, self.norm//2, float(self.norm)
            x = torch.clamp(round_func((pred / avg_factor) + bias), 0, self.norm) / norm
        else:
            # Stage 1
            # reduce self.stages, only one feature stage
            for s in range(self.stages-1):
                pred = 0
                for mode in self.modes:
                    pad = mode_pad_dict[mode]
                    for r in [0, 1, 2, 3]:
                        pred += round_func(torch.rot90(self.forward(F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), stage=s+1, mode=mode, r=0), (4 - r) % 4, [2, 3]) * (self.norm//2))
                if (s+1 == self.stages-1):
                    avg_factor, bias, norm = len(self.modes), 0, 1
                else:
                    avg_factor, bias, norm = len(self.modes) * 4, self.norm//2, float(self.norm)
                x = torch.clamp(round_func((pred / avg_factor)) + bias, 0, self.norm) / norm

        return x


class SWF2LUT(nn.Module):
    def __init__(self, opt, inC=1, outC=3):
        super(SWF2LUT, self).__init__()
        self.modes2 = opt.modes2
        self.modes = opt.modes
        self.stages = opt.stages
        self.norm = opt.norm
        self.interval = opt.interval

        # hyper stage
        stage = 2
        self.outC = outC
        for mode in self.modes2:
            for r in [0, 1]:
                key = "s{}_{}r{}".format(str(stage), mode, r)
                lut_path = os.path.join(opt.expDir, "LUT_{}.npy".format(key))
                # self.luts[key] = torch.Tensor(np.load(lut_path)).cuda()
                lut_arr = np.load(lut_path).reshape(-1, outC).astype(np.float32)/127.0
                self.register_parameter(name="weight_"+key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))
                print(lut_path, lut_arr.shape)

        stage = 1
        for mode in self.modes:
            key = "s{}_{}r{}".format(str(stage), mode, 0)
            lut_path = os.path.join(opt.expDir, "LUT_{}.npy".format(key))
            # self.luts[key] = torch.Tensor(np.load(lut_path)).cuda()
            lut_arr = np.load(lut_path).reshape(-1, 1).astype(np.float32)/127.0
            self.register_parameter(name="weight_"+key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))
            print(lut_path, lut_arr.shape)


    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable) with an identity function (differentiable) only when backward,
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight, outC, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2**interval
        L = 2**(8-interval) + 1
        
        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_a1 = torch.floor_divide(img_in[:, :, 0:0+h, 0:0+w],q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0+h, 1:1+w],q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 1:1+h, 0:0+w],q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 1:1+h, 1:1+w],q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0+h, 0:0+w] % q
            fb = img_in[:, :, 0:0+h, 1:1+w] % q
            fc = img_in[:, :, 1:1+h, 0:0+w] % q
            fd = img_in[:, :, 1:1+h, 1:1+w] % q
        
        elif mode == "d":         
            img_a1 = torch.floor_divide(img_in[:, :, 0:0+h, 0:0+w],q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0+h, 2:2+w],q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 2:2+h, 0:0+w],q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 2:2+h, 2:2+w],q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0+h, 0:0+w] % q
            fb = img_in[:, :, 0:0+h, 2:2+w] % q
            fc = img_in[:, :, 2:2+h, 0:0+w] % q
            fd = img_in[:, :, 2:2+h, 2:2+w] % q
        
        elif mode == "y":         
            img_a1 = torch.floor_divide(img_in[:, :, 0:0+h, 0:0+w],q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 1:1+h, 1:1+w],q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 1:1+h, 2:2+w],q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 2:2+h, 1:1+w],q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0+h, 0:0+w] % q
            fb = img_in[:, :, 1:1+h, 1:1+w] % q
            fc = img_in[:, :, 1:1+h, 2:2+w] % q
            fd = img_in[:, :, 2:2+h, 1:1+w] % q      
        elif mode == "c":         
            img_a1 = torch.floor_divide(img_in[:, :, 0:0+h, 0:0+w],q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0+h, 1:1+w],q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 0:0+h, 2:2+w],q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 0:0+h, 3:3+w],q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0+h, 0:0+w] % q
            fb = img_in[:, :, 1:1+h, 1:1+w] % q
            fc = img_in[:, :, 1:1+h, 2:2+w] % q
            fd = img_in[:, :, 2:2+h, 1:1+w] % q    
        elif mode == "t":         
            img_a1 = torch.floor_divide(img_in[:, :, 0:0+h, 0:0+w],q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 1:1+h, 1:1+w],q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 2:2+h, 2:2+w],q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 3:3+h, 3:3+w],q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0+h, 0:0+w] % q
            fb = img_in[:, :, 1:1+h, 1:1+w] % q
            fc = img_in[:, :, 1:1+h, 2:2+w] % q
            fd = img_in[:, :, 2:2+h, 1:1+w] % q   
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode)) 
        

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1


        p0000 = weight[img_a1.flatten()*L*L*L + img_b1.flatten()*L*L + img_c1.flatten()*L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p0001 = weight[img_a1.flatten()*L*L*L + img_b1.flatten()*L*L + img_c1.flatten()*L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p0010 = weight[img_a1.flatten()*L*L*L + img_b1.flatten()*L*L + img_c2.flatten()*L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p0011 = weight[img_a1.flatten()*L*L*L + img_b1.flatten()*L*L + img_c2.flatten()*L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p0100 = weight[img_a1.flatten()*L*L*L + img_b2.flatten()*L*L + img_c1.flatten()*L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p0101 = weight[img_a1.flatten()*L*L*L + img_b2.flatten()*L*L + img_c1.flatten()*L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p0110 = weight[img_a1.flatten()*L*L*L + img_b2.flatten()*L*L + img_c2.flatten()*L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p0111 = weight[img_a1.flatten()*L*L*L + img_b2.flatten()*L*L + img_c2.flatten()*L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))

        p1000 = weight[img_a2.flatten()*L*L*L + img_b1.flatten()*L*L + img_c1.flatten()*L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p1001 = weight[img_a2.flatten()*L*L*L + img_b1.flatten()*L*L + img_c1.flatten()*L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p1010 = weight[img_a2.flatten()*L*L*L + img_b1.flatten()*L*L + img_c2.flatten()*L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p1011 = weight[img_a2.flatten()*L*L*L + img_b1.flatten()*L*L + img_c2.flatten()*L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p1100 = weight[img_a2.flatten()*L*L*L + img_b2.flatten()*L*L + img_c1.flatten()*L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p1101 = weight[img_a2.flatten()*L*L*L + img_b2.flatten()*L*L + img_c1.flatten()*L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p1110 = weight[img_a2.flatten()*L*L*L + img_b2.flatten()*L*L + img_c2.flatten()*L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        p1111 = weight[img_a2.flatten()*L*L*L + img_b2.flatten()*L*L + img_c2.flatten()*L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                          img_a1.shape[3], outC), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0]*img_a1.shape[1]*img_a1.shape[2]*img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa>fb; fac = fa>fc; fad = fa>fd

        fbc = fb>fc; fbd = fb>fd; fcd = fc>fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q-fa[i]) * p0000[i] + (fa[i]-fb[i]) * p1000[i] + (fb[i]-fc[i]) * p1100[i] + (fc[i]-fd[i]) * p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:,None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q-fa[i]) * p0000[i] + (fa[i]-fb[i]) * p1000[i] + (fb[i]-fd[i]) * p1100[i] + (fd[i]-fc[i]) * p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:,None], ~i2[:,None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q-fa[i]) * p0000[i] + (fa[i]-fd[i]) * p1000[i] + (fd[i]-fb[i]) * p1001[i] + (fb[i]-fc[i]) * p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:,None], ~i2[:,None], ~i3[:,None], fab, fbc], dim=1), dim=1)
        out[i] = (q-fd[i]) * p0000[i] + (fd[i]-fa[i]) * p0001[i] + (fa[i]-fb[i]) * p1001[i] + (fb[i]-fc[i]) * p1101[i] + (fc[i]) * p1111[i]
        
        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q-fa[i]) * p0000[i] + (fa[i]-fc[i]) * p1000[i] + (fc[i]-fb[i]) * p1010[i] + (fb[i]-fd[i]) * p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:,None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q-fa[i]) * p0000[i] + (fa[i]-fc[i]) * p1000[i] + (fc[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:,None], ~i6[:,None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q-fa[i]) * p0000[i] + (fa[i]-fd[i]) * p1000[i] + (fd[i]-fc[i]) * p1001[i] + (fc[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:,None], ~i6[:,None], ~i7[:,None], fab, fac], dim=1), dim=1)
        out[i] = (q-fd[i]) * p0000[i] + (fd[i]-fa[i]) * p0001[i] + (fa[i]-fc[i]) * p1001[i] + (fc[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        
        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fb[i]) * p1010[i] + (fb[i]-fd[i]) * p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fad], dim=1), dim=1) # c > a > d > b
        out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fcd], dim=1), dim=1) # c > d > a > b
        out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], ~i11[:,None], fab], dim=1), dim=1)
        out[i] = (q-fd[i]) * p0000[i] + (fd[i]-fc[i]) * p0001[i] + (fc[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i] 
        
        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q-fb[i]) * p0000[i] + (fb[i]-fa[i]) * p0100[i] + (fa[i]-fc[i]) * p1100[i] + (fc[i]-fd[i]) * p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:,None], fac, fad], dim=1), dim=1)
        out[i] = (q-fb[i]) * p0000[i] + (fb[i]-fa[i]) * p0100[i] + (fa[i]-fd[i]) * p1100[i] + (fd[i]-fc[i]) * p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:,None], ~i14[:,None], fac, fbd], dim=1), dim=1)
        out[i] = (q-fb[i]) * p0000[i] + (fb[i]-fd[i]) * p0100[i] + (fd[i]-fa[i]) * p0101[i] + (fa[i]-fc[i]) * p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:,None], ~i14[:,None], ~i15[:,None], fac], dim=1), dim=1)
        out[i] = (q-fd[i]) * p0000[i] + (fd[i]-fb[i]) * p0001[i] + (fb[i]-fa[i]) * p0101[i] + (fa[i]-fc[i]) * p1101[i] + (fc[i]) * p1111[i]
        
        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q-fb[i]) * p0000[i] + (fb[i]-fc[i]) * p0100[i] + (fc[i]-fa[i]) * p0110[i] + (fa[i]-fd[i]) * p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:,None], fbc, fcd], dim=1), dim=1)
        out[i] = (q-fb[i]) * p0000[i] + (fb[i]-fc[i]) * p0100[i] + (fc[i]-fd[i]) * p0110[i] + (fd[i]-fa[i]) * p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:,None], ~i18[:,None], fbc, fbd], dim=1), dim=1)
        out[i] = (q-fb[i]) * p0000[i] + (fb[i]-fd[i]) * p0100[i] + (fd[i]-fc[i]) * p0101[i] + (fc[i]-fa[i]) * p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:,None], ~i18[:,None], ~i19[:,None], fbc], dim=1), dim=1)
        out[i] = (q-fd[i]) * p0000[i] + (fd[i]-fb[i]) * p0001[i] + (fb[i]-fc[i]) * p0101[i] + (fc[i]-fa[i]) * p0111[i] + (fa[i]) * p1111[i]
        
        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fb[i]) * p0010[i] + (fb[i]-fa[i]) * p0110[i] + (fa[i]-fd[i]) * p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:,None], fbd], dim=1), dim=1)
        out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fb[i]) * p0010[i] + (fb[i]-fd[i]) * p0110[i] + (fd[i]-fa[i]) * p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:,None], ~i22[:,None], fcd], dim=1), dim=1)
        out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fb[i]) * p0011[i] + (fb[i]-fa[i]) * p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:,None], ~i22[:,None], ~i23[:,None]], dim=1), dim=1)
        out[i] = (q-fd[i]) * p0000[i] + (fd[i]-fc[i]) * p0001[i] + (fc[i]-fb[i]) * p0011[i] + (fb[i]-fa[i]) * p0111[i] + (fa[i]) * p1111[i]        
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        # out = out.permute(0, 1, 2,4, 3,5).reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]*upscale, img_a1.shape[3]*upscale))
        # out = out.permute(0, 1, 2,4, 3,5).reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], outC))
        out = out.permute(0, 1, 4, 2, 3).reshape((img_a1.shape[0], img_a1.shape[1]*outC, img_a1.shape[2], img_a1.shape[3]))
        out = out / q
        return out

    def forward(self, x, stage, mode, r):
        key = "s{}_{}r{}".format(str(stage), mode, r)
        pad = mode_pad_dict[mode]
        if stage == 1:
            outC = 1
        else:
            outC = self.outC
        weight = getattr(self, "weight_"+key)
        result = self.InterpTorchBatch(weight, outC, mode, x, pad)
        return result

    def predict(self, x, stage=None):
        # 8bit input
        x = round_func(x * 255.0)
        
        # LUT no need to multiply self.norm//2
        if stage == 2: # hyper stage
            pred = 0
            for mode in self.modes2:
                pad = mode_pad_dict[mode]
                for r in [0, 2]:
                    pred += round_func(torch.rot90(self.forward(F.pad(torch.rot90(x, r, [
                        2, 3]), (0, pad, 0, pad), mode='replicate'), stage=self.stages, mode=mode, r=0), (4 - r) % 4, [2, 3])) # * (self.norm//2))
                for r in [1, 3]:
                    pred += round_func(torch.rot90(self.forward(F.pad(torch.rot90(x, r, [
                        2, 3]), (0, pad, 0, pad), mode='replicate'), stage=self.stages, mode=mode, r=1), (4 - r) % 4, [2, 3])) # * (self.norm//2))
            avg_factor, bias, norm = len(self.modes2) * 4, self.norm//2, float(self.norm)
            x = torch.clamp(round_func((pred / avg_factor) + bias), 0, self.norm) / norm
        else:
            # Stage 1
            # reduce self.stages, only one feature stage
            for s in range(self.stages-1):
                pred = 0
                for mode in self.modes:
                    pad = mode_pad_dict[mode]
                    for r in [0, 1, 2, 3]:
                        pred += round_func(torch.rot90(self.forward(F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), stage=s+1, mode=mode, r=0), (4 - r) % 4, [2, 3])) # * (self.norm//2))
                if (s+1 == self.stages-1):
                    avg_factor, bias, norm = len(self.modes), 0, 1
                else:
                    avg_factor, bias, norm = len(self.modes) * 4, self.norm//2, float(self.norm)
                x = torch.clamp(round_func((pred / avg_factor)) + bias, 0, self.norm) / norm

        return x


# https://github.com/Zheng222/IMDN/blob/master/model/block.py
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class IMDModule_speed(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, 1)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused

# https://github.com/Zheng222/IMDN/blob/master/model/architecture.py
# AI in RTC Image Super-Resolution Algorithm Performance Comparison Challenge (Winner solution)
class IMDN_RTC(nn.Module):
    def __init__(self, in_nc=3, nf=12, num_modules=5, out_nc=3, upscale=2):
        super(IMDN_RTC, self).__init__()

        fea_conv = [conv_layer(in_nc, nf, kernel_size=3)]
        rb_blocks = [IMDModule_speed(in_channels=nf) for _ in range(num_modules)]
        LR_conv = conv_layer(nf, nf, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output
    

class IMDN2(nn.Module):
    def __init__(self, opt, inC=1, outC=1):
        super(IMDN2, self).__init__()
        self.norm = opt.norm
        self.stage1 = IMDN_RTC(nf=opt.nf, in_nc=inC, out_nc=inC, upscale=1)
        self.stage2 = IMDN_RTC(nf=opt.nf, in_nc=inC, out_nc=inC*outC, upscale=1)

    def predict(self, x, stage=1):
        if stage == 2: # hyper: [0-1]
            return (torch.clamp(self.stage2(x), -1, 1)/2 + 1/2)
        else:
            return (torch.clamp(self.stage1(x), -1, 1) * (self.norm//2) + (self.norm//2))
        