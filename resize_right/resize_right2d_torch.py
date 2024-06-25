from math import ceil
import torch

import sys
sys.path.insert(0, "./")

import resize_right.interp_methods as interp_methods

class Resize2dTorch(object):
    def __init__(self, support_sz=4, device="GPU", pad_mode='constant'):
        self.eps = torch.finfo(torch.float32).eps
        self.device = device
        self.init_support_sz = support_sz
        self.pad_mode = pad_mode
        
        self.antialias = False

    def set_shape(self, in_shape, scale_factors=None, out_shape=None):
        self.support_sz = self.init_support_sz
        self.in_shape = in_shape
        self.set_scale_and_out_sz(in_shape, scale_factors, out_shape)
        self.get_distance(self.in_shape, self.out_shape, self.scale_factors)
        

    def set_scale_and_out_sz(self, in_shape, scale_factors, out_shape):
        if out_shape is not None:
            out_shape = (list(in_shape[:-len(out_shape)]) + list(out_shape))
            if scale_factors is None:
                scale_factors = [out_sz /in_sz for out_sz, in_sz in zip(out_shape, in_shape)]
        if scale_factors is not None:
            scale_factors = (scale_factors if isinstance(scale_factors, (list, tuple))
                                            else [scale_factors, scale_factors])
            scale_factors = ([1] * (len(in_shape) - len(scale_factors)) + list(scale_factors))
            if out_shape is None:
                out_shape = [ceil(scale_factor * in_sz) for scale_factor, in_sz in zip(scale_factors, in_shape)]
        self.scale_factors = [float(s) for s in scale_factors]
        self.out_shape = out_shape
        
        # apply anti-aliasing for downsampling
        # print(scale_factors)
        # if scale_factors[2] < 1.0 or scale_factors[3] < 1.0:
        if False:
            self.antialias = True
            self.min_scale_factor = min([scale_factors[2], scale_factors[3]])
            self.support_sz = ceil(self.support_sz / self.min_scale_factor)
            # print(self.min_scale_factor, self.support_sz)

    def get_projected_grid2d(self, scale_factor, in_shape, out_shape):
        in_sz = [in_shape[2], in_shape[3]]
        out_sz = [out_shape[2], out_shape[3]]
        grid_sz_h = out_sz[0]
        grid_sz_w = out_sz[1]
        x = torch.arange(grid_sz_h, device=self.device)
        y = torch.arange(grid_sz_w, device=self.device)

        x_r = torch.repeat_interleave(x, self.support_sz) 
        y_r = torch.repeat_interleave(y, self.support_sz) 

        # torch verison 1.8.0
        X,Y = torch.meshgrid(x_r, y_r, indexing="ij") # in 1.11, add indexing="ij"
        grid_x = (X /float(scale_factor[2]) + (in_sz[0] - 1) / 2 - (out_sz[0] - 1) /(2*float(scale_factor[2])))
        grid_y = (Y /float(scale_factor[3]) + (in_sz[1] - 1) / 2 - (out_sz[1] - 1) /(2*float(scale_factor[3])))
        return grid_x, grid_y

    def get_field_of_view2d(self, projected_grid_x, projected_grid_y, out_shape):
        out_sz = [out_shape[2], out_shape[3]]
        cur_support_sz = self.support_sz
        left_boundaries_x = (projected_grid_x - cur_support_sz / 2 -self.eps).ceil().long()
        left_boundaries_y = (projected_grid_y - cur_support_sz / 2 -self.eps).ceil().long()

        ordinal_numbers_x = torch.arange(ceil(cur_support_sz - self.eps), device=self.device)
        ordinal_numbers_y = torch.arange(ceil(cur_support_sz - self.eps), device=self.device)

        ord_X, ord_Y = torch.meshgrid(ordinal_numbers_x, ordinal_numbers_y, indexing="ij")
        ord_X_r = ord_X.repeat(out_sz[0], out_sz[1])
        ord_Y_r = ord_Y.repeat(out_sz[0], out_sz[1])

        return left_boundaries_x + ord_X_r, left_boundaries_y + ord_Y_r

    def cal_pad_sz(self, in_sz, out_sz, field_of_view, projected_grid, scale_factor):
        pad_sz = [-field_of_view[0, 0].item(), field_of_view[-1, -1].item() - in_sz + 1]
        field_of_view += pad_sz[0]
        projected_grid += pad_sz[0]
        return pad_sz, projected_grid, field_of_view

    def get_distance(self, in_shape, out_shape, scale_factors):
        projected_grid_x, projected_grid_y = self.get_projected_grid2d(scale_factors, in_shape, out_shape)
        field_of_view_x, field_of_view_y = self.get_field_of_view2d(projected_grid_x, projected_grid_y, out_shape)

        pad_sz_x, projected_grid_x, field_of_view_x = self.cal_pad_sz(in_shape[2], out_shape[2], field_of_view_x, projected_grid_x, self.scale_factors[0])
        pad_sz_y, projected_grid_y, field_of_view_y = self.cal_pad_sz(in_shape[3], out_shape[3], field_of_view_y, projected_grid_y, self.scale_factors[1])

        pad_vec = [0] * 4
        pad_vec[0:2] = pad_sz_x
        pad_vec[2:4] = pad_sz_y

        self.pad_vec = pad_vec

        dis_x, dis_y = projected_grid_x-field_of_view_x, projected_grid_y-field_of_view_y

        self.field_of_view_x, self.field_of_view_y = field_of_view_x, field_of_view_y
        self.dis_x = torch.cat([torch.unsqueeze(torch.unsqueeze(dis_x, 0), 0)] * in_shape[0], dim=0)
        self.dis_y = torch.cat([torch.unsqueeze(torch.unsqueeze(dis_y, 0), 0)] * in_shape[0], dim=0)

    def resize(self, input):
        # default interpolation warping process
        B, C, H, W = input.shape
        support_sz = self.support_sz
        
        weights = self.weight(self.dis_x, self.dis_y)
        # to C channels
        weights = torch.cat([weights]*C, dim=1)
        # normalize
        weights = torch.reshape(weights, (B, C, self.out_sz[0], support_sz, self.out_sz[1], support_sz))
        weights = weights.transpose(3, 4)
        weights = torch.reshape(weights, (B, C, self.out_sz[0], self.out_sz[1], support_sz*support_sz))
        if support_sz != 1:
            weights_patch_sum = weights.sum(axis=4, keepdim=True)
            weights = weights / weights_patch_sum

        tmp_input = torch.nn.functional.pad(input, pad=self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[:, :, self.field_of_view_x, self.field_of_view_y] # [B, C, outH*sz, outW*sz]
        neighbors = torch.reshape(neighbors, (B, C, self.out_sz[0], support_sz, self.out_sz[1], support_sz))
        neighbors = neighbors.transpose(3, 4)
        neighbors = torch.reshape(neighbors, (B, C, self.out_sz[0], self.out_sz[1], support_sz*support_sz))

        output = neighbors * weights
        output = output.sum(4)
        return output
    
class BicubicResize2dTorch(Resize2dTorch):
    def __init__(self, support_sz=4, device="CPU", pad_mode="constant"):
        super().__init__(support_sz, device, pad_mode)
        self.support_sz = support_sz
        self.antialias = False
        
    def weight(self, x, y):
        return interp_methods.cubic2d(x, y)

class SteeringGaussianResize2dTorch(Resize2dTorch):
    def __init__(self, support_sz=4, device="GPU", pad_mode='constant', max_sigma=10):
        super().__init__(support_sz, device, pad_mode)
        self.max_sigma = max_sigma
    
    def sk_weight(self, rho, sigma_x, sigma_y, x, y):
        multiplier = 1 # 1 / (2*pi*sigma_x*sigma_y*troch.sqrt(1-rho**2) + self.eps)
        e_multiplier = -1/2 # 1/(self.max_sigma) # -1 * (1/(2*(1-rho**2)+self.eps))
        x_nominal = (sigma_x * x) ** 2
        y_nominal = (sigma_y * y) ** 2
        xy_nominal = sigma_x * x * sigma_y * y
        result = multiplier * torch.exp(e_multiplier * (x_nominal - 2*rho*xy_nominal + y_nominal))
        return result

    def resize(self, input, rho, sigma_x, sigma_y):
        B, C, H, W = input.shape
        out_sz = [self.out_shape[2], self.out_shape[3]]
        support_sz = self.support_sz

        rho = rho * 2 - 1
        sigma_x = sigma_x * self.max_sigma
        sigma_y = sigma_y * self.max_sigma

        tmp_rho = torch.nn.functional.pad(rho, pad=self.pad_vec, mode='replicate')
        tmp_simga_x = torch.nn.functional.pad(sigma_x, pad=self.pad_vec, mode='replicate')
        tmp_simga_y = torch.nn.functional.pad(sigma_y, pad=self.pad_vec, mode='replicate')

        # avoid possible out of bound CUDA Runtime error, only the max index exceed one pixel
        self.field_of_view_x = self.field_of_view_x.clip(0, tmp_rho.shape[-2]-1)
        self.field_of_view_y = self.field_of_view_y.clip(0, tmp_rho.shape[-1]-1)
        factor_rho = tmp_rho[:, :, self.field_of_view_x, self.field_of_view_y]
        factor_sigma_x = tmp_simga_x[:, :, self.field_of_view_x, self.field_of_view_y]
        factor_sigma_y = tmp_simga_y[:, :, self.field_of_view_x, self.field_of_view_y]

        if self.antialias:
            # print(self.support_sz, input.shape)
            # print(self.field_of_view_x.shape)
            weights = self.min_scale_factor * self.sk_weight(factor_rho, factor_sigma_x, factor_sigma_y, self.min_scale_factor * self.dis_x, self.min_scale_factor * self.dis_y)
            # print("Anti-Aliasing")
        else:
            weights = self.sk_weight(factor_rho, factor_sigma_x, factor_sigma_y, self.dis_x, self.dis_y)

        # normalize
        weights = torch.reshape(weights, (B, C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = weights.transpose(3, 4)
        weights = torch.reshape(weights, (B, C, out_sz[0], out_sz[1], support_sz*support_sz))
        weights_patch_sum = weights.sum(axis=4, keepdim=True)
        weights = weights / weights_patch_sum

        tmp_input = torch.nn.functional.pad(input, pad=self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[:, :, self.field_of_view_x, self.field_of_view_y] # [B, C, outH*sz, outW*sz]
        neighbors = torch.reshape(neighbors, (B, C, out_sz[0], support_sz, out_sz[1], support_sz))
        neighbors = neighbors.transpose(3, 4)
        neighbors = torch.reshape(neighbors, (B, C, out_sz[0], out_sz[1], support_sz*support_sz))

        output = neighbors * weights
        output = output.sum(4)
        return output
    
class AmplifiedLinearResize2dTorch(Resize2dTorch):
    def __init__(self, support_sz=2, device="GPU", pad_mode='constant', max_sigma=1):
        super().__init__(support_sz, device, pad_mode)
        self.max_sigma = max_sigma

    def linear_alpha(self, x, alpha):
        return ((alpha * x + 1) * ((-1 <= x) & (x < 0)) + (1 - alpha * x) *
                ((0 <= x) & (x <= 1)))

    def linear_weight(self, alpha, x, y):
        # weight = linear(x) * linear(y)
        # negative weights are clipped
        weight = torch.clamp(self.linear_alpha(x, alpha), 0, None) * torch.clamp(self.linear_alpha(y, alpha), 0, None)
        return weight

    def resize(self, input, alpha):
        B, C, H, W = input.shape
        out_sz = [self.out_shape[2], self.out_shape[3]]
        support_sz = self.support_sz

        # [0, 1] to [0, max_sigma]
        # alpha = (alpha) * self.max_sigma # sigma amplification leads to nan loss (near zero weights)
        # [0, 1] to [-max_sigma, max_sigma]
        alpha = alpha * 2 - 1
        alpha = self.max_sigma * alpha
        # alpha = torch.exp(alpha)
        tmp_alpha = torch.nn.functional.pad(alpha, pad=self.pad_vec, mode='replicate')

        factor_alpha = tmp_alpha[:, :, self.field_of_view_x, self.field_of_view_y]

        weights = self.linear_weight(factor_alpha, self.dis_x, self.dis_y)

        # normalize
        weights = torch.reshape(weights, (B, C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = weights.transpose(3, 4)
        weights = torch.reshape(weights, (B, C, out_sz[0], out_sz[1], support_sz*support_sz))
        # print(weights[0, 0, 0, :4])
        weights_patch_sum = weights.sum(axis=4, keepdim=True)
        weights = weights / weights_patch_sum

        tmp_input = torch.nn.functional.pad(input, pad=self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[:, :, self.field_of_view_x, self.field_of_view_y] # [B, C, outH*sz, outW*sz]
        neighbors = torch.reshape(neighbors, (B, C, out_sz[0], support_sz, out_sz[1], support_sz))
        neighbors = neighbors.transpose(3, 4)
        neighbors = torch.reshape(neighbors, (B, C, out_sz[0], out_sz[1], support_sz*support_sz))

        output = neighbors * weights
        output = output.sum(4)
        return output

class Warp2dTorch(object):
    def __init__(self, support_sz=4, device="GPU", pad_mode='constant'):
        self.eps = torch.finfo(torch.float32).eps
        self.device = device
        self.support_sz = support_sz
        self.pad_mode = pad_mode

    def set_shape(self, in_shape, matrix, out_shape):
        self.in_shape = in_shape
        self.matrix = matrix
        self.set_scale_and_out_sz(in_shape, matrix, out_shape)
        self.get_distance(self.matrix)

    def set_scale_and_out_sz(self, in_shape, matrix, out_shape):
        out_shape = list(out_shape) + list(in_shape[len(out_shape) :])
        self.out_shape = out_shape
        self.in_sz = [in_shape[2], in_shape[3]]
        self.out_sz = [out_shape[2], out_shape[3]]

    def get_projected_grid2d(self, matrix):
        grid_sz_h = self.out_sz[0]
        grid_sz_w = self.out_sz[1]
        x = torch.arange(grid_sz_h, device=self.device)
        y = torch.arange(grid_sz_w, device=self.device)

        x_r = torch.repeat_interleave(x, self.support_sz) 
        y_r = torch.repeat_interleave(y, self.support_sz) 

        # gx, gy = torch.meshgrid(x_r, y_r) # torch verison 1.8.0
        # ret = torch.stack([gx, gy], dim=-1)
        grid = torch.meshgrid(x_r, y_r, indexing="ij") # in 1.11, add indexing="ij"
        ret = torch.stack(grid, dim=-1)
        gridy = ret.reshape(-1, ret.shape[-1]) #.astype(np.float32)
        
        # h -> y, w -> x
        gridy = gridy.flip(-1)
        # linalg.inv requires double()
        gridx = torch.mm(torch.linalg.inv(matrix), torch.cat([gridy, torch.ones([gridy.shape[0], 1], device=self.device).double()], dim=-1).permute(1, 0)).permute(1, 0)   
        gridx[:, 0] /= gridx[:, -1]
        gridx[:, 1] /= gridx[:, -1]
        gridx = gridx[:, 0:2]     
        # reverse back
        gridx = gridx.flip(-1)

        # clipped out of field coordinates
        grid_x = gridx[:, 0].reshape(grid_sz_h*self.support_sz, grid_sz_w*self.support_sz).clip(0, self.in_sz[0])
        grid_y = gridx[:, 1].reshape(grid_sz_h*self.support_sz, grid_sz_w*self.support_sz).clip(0, self.in_sz[1])
        return grid_x.double(), grid_y.double()

    def get_field_of_view2d(self, projected_grid_x, projected_grid_y):
        cur_support_sz = self.support_sz
        left_boundaries_x = (projected_grid_x - cur_support_sz / 2 -self.eps).ceil().long()
        left_boundaries_y = (projected_grid_y - cur_support_sz / 2 -self.eps).ceil().long()

        ordinal_numbers_x = torch.arange(ceil(cur_support_sz - self.eps), device=self.device)
        ordinal_numbers_y = torch.arange(ceil(cur_support_sz - self.eps), device=self.device)

        # torch 1.8.0
        # ord_X, ord_Y = torch.meshgrid(ordinal_numbers_x, ordinal_numbers_y)
        # ord_X = ord_X.transpose(1, 0)
        # ord_Y = ord_Y.transpose(1, 0)
        ord_X, ord_Y = torch.meshgrid(ordinal_numbers_x, ordinal_numbers_y, indexing="ij")
        ord_X_r = ord_X.repeat(self.out_sz[0], self.out_sz[1])
        ord_Y_r = ord_Y.repeat(self.out_sz[0], self.out_sz[1])
        return left_boundaries_x + ord_X_r, left_boundaries_y + ord_Y_r

    def cal_pad_sz(self, in_sz, field_of_view, projected_grid):
        # prevent negative pad_width
        pad_sz = (max(-field_of_view[0, 0].item(), 0), max(field_of_view[-1, -1].item() - in_sz + 1, 0))
        # pad_sz = (-field_of_view[0, 0].item(), field_of_view[-1, -1].item() - self.in_sz + 1)
        field_of_view += pad_sz[0]
        projected_grid += pad_sz[0]
        return pad_sz, projected_grid, field_of_view

    def get_distance(self, matrix):
        projected_grid_x, projected_grid_y = self.get_projected_grid2d(matrix)
        field_of_view_x, field_of_view_y = self.get_field_of_view2d(projected_grid_x, projected_grid_y)

        pad_sz_x, projected_grid_x, field_of_view_x = self.cal_pad_sz(self.in_sz[0], field_of_view_x, projected_grid_x)
        pad_sz_y, projected_grid_y, field_of_view_y = self.cal_pad_sz(self.in_sz[1], field_of_view_y, projected_grid_y)

        pad_vec = [0] * 4
        pad_vec[0:2] = pad_sz_y # last dim first, different from np.pad
        pad_vec[2:4] = pad_sz_x

        self.pad_vec = pad_vec
        self.projected_grid_x, self.projected_grid_y = projected_grid_x, projected_grid_y
        self.field_of_view_x, self.field_of_view_y = field_of_view_x, field_of_view_y

        # clip out of field pixels
        self.field_of_view_x = field_of_view_x.clip(0, self.in_sz[0]-1)
        self.field_of_view_y = field_of_view_y.clip(0, self.in_sz[1]-1)

        dis_x, dis_y = self.projected_grid_x-self.field_of_view_x, self.projected_grid_y-self.field_of_view_y
        self.dis_x = torch.cat([torch.unsqueeze(torch.unsqueeze(dis_x, 0), 0)] * self.in_shape[0], dim=0)
        self.dis_y = torch.cat([torch.unsqueeze(torch.unsqueeze(dis_y, 0), 0)] * self.in_shape[0], dim=0)

    def warp(self, input):
        # default interpolation warping process
        B, C, H, W = input.shape
        support_sz = self.support_sz
        
        weights = self.weight(self.dis_x, self.dis_y)
        # to C channels
        weights = torch.cat([weights]*C, dim=1)
        # normalize
        weights = torch.reshape(weights, (B, C, self.out_sz[0], support_sz, self.out_sz[1], support_sz))
        weights = weights.transpose(3, 4)
        weights = torch.reshape(weights, (B, C, self.out_sz[0], self.out_sz[1], support_sz*support_sz))
        if support_sz != 1:
            weights_patch_sum = weights.sum(axis=4, keepdim=True)
            weights = weights / weights_patch_sum

        tmp_input = torch.nn.functional.pad(input, pad=self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[:, :, self.field_of_view_x, self.field_of_view_y] # [B, C, outH*sz, outW*sz]
        neighbors = torch.reshape(neighbors, (B, C, self.out_sz[0], support_sz, self.out_sz[1], support_sz))
        neighbors = neighbors.transpose(3, 4)
        neighbors = torch.reshape(neighbors, (B, C, self.out_sz[0], self.out_sz[1], support_sz*support_sz))

        output = neighbors * weights
        output = output.sum(4)
        return output
    
class BicubicWarp2dTorch(Warp2dTorch):
    def __init__(self, support_sz=4, device="CPU", pad_mode="constant"):
        super().__init__(support_sz, device, pad_mode)
        self.support_sz = support_sz
        self.antialias = False
        
    def weight(self, x, y):
        return interp_methods.cubic2d(x, y)
    
class NearestWarp2dTorch(Warp2dTorch):
    def __init__(self, support_sz=1, device="CPU", pad_mode="constant"):
        super().__init__(support_sz, device, pad_mode)
        self.support_sz = support_sz
        self.antialias = False
        
    def weight(self, x, y):
        return interp_methods.box2d(x, y)

class SteeringGaussianWarp2dTorch(Warp2dTorch):
    def __init__(self, support_sz=4, device="GPU", pad_mode='constant', max_sigma=10):
        super().__init__(support_sz, device, pad_mode)
        self.max_sigma = max_sigma
    
    def sk_weight(self, rho, sigma_x, sigma_y, x, y):
        multiplier = 1 # 1 / (2*pi*sigma_x*sigma_y*troch.sqrt(1-rho**2) + self.eps)
        e_multiplier = -1/2 # 1/(self.max_sigma) # -1 * (1/(2*(1-rho**2)+self.eps))
        x_nominal = (sigma_x * x) ** 2
        y_nominal = (sigma_y * y) ** 2
        xy_nominal = sigma_x * x * sigma_y * y
        result = multiplier * torch.exp(e_multiplier * (x_nominal - 2*rho*xy_nominal + y_nominal))
        return result

    def warp(self, input, rho, sigma_x, sigma_y):
        B, C, H, W = input.shape
        out_sz = self.out_sz
        support_sz = self.support_sz

        rho = rho * 2 - 1
        sigma_x = sigma_x * self.max_sigma
        sigma_y = sigma_y * self.max_sigma

        tmp_rho = torch.nn.functional.pad(rho, pad=self.pad_vec, mode='replicate')
        tmp_simga_x = torch.nn.functional.pad(sigma_x, pad=self.pad_vec, mode='replicate')
        tmp_simga_y = torch.nn.functional.pad(sigma_y, pad=self.pad_vec, mode='replicate')

        factor_rho = tmp_rho[:, :, self.field_of_view_x, self.field_of_view_y]
        factor_sigma_x = tmp_simga_x[:, :, self.field_of_view_x, self.field_of_view_y]
        factor_sigma_y = tmp_simga_y[:, :, self.field_of_view_x, self.field_of_view_y]

        weights = self.sk_weight(factor_rho, factor_sigma_x, factor_sigma_y, self.dis_x, self.dis_y)

        # normalize
        weights = torch.reshape(weights, (B, C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = weights.transpose(3, 4)
        weights = torch.reshape(weights, (B, C, out_sz[0], out_sz[1], support_sz*support_sz))
        weights_patch_sum = weights.sum(axis=4, keepdim=True)
        weights = weights / weights_patch_sum

        tmp_input = torch.nn.functional.pad(input, pad=self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[:, :, self.field_of_view_x, self.field_of_view_y] # [B, C, outH*sz, outW*sz]
        neighbors = torch.reshape(neighbors, (B, C, out_sz[0], support_sz, out_sz[1], support_sz))
        neighbors = neighbors.transpose(3, 4)
        neighbors = torch.reshape(neighbors, (B, C, out_sz[0], out_sz[1], support_sz*support_sz))

        output = neighbors * weights
        output = output.sum(4)
        return output
    
class AmplifiedLinearWarp2dTorch(Warp2dTorch):
    def __init__(self, support_sz=2, device="GPU", pad_mode='constant', max_sigma=1):
        super().__init__(support_sz, device, pad_mode)
        self.max_sigma = max_sigma
    
    def linear_alpha(self, x, alpha):
        return ((alpha * x + 1) * ((-1 <= x) & (x < 0)) + (1 - alpha * x) *
                ((0 <= x) & (x <= 1)))

    def linear_weight(self, alpha, x, y):
        # weight = linear(x) * linear(y)
        # negative weights are clipped
        weight = torch.clamp(self.linear_alpha(x, alpha), 0, None) * torch.clamp(self.linear_alpha(y, alpha), 0, None)
        return weight

    def warp(self, input, alpha):
        B, C, H, W = input.shape
        out_sz = self.out_sz
        support_sz = self.support_sz

        # [0, 1] to [0, max_sigma]
        # alpha = (alpha) * self.max_sigma # sigma amplification leads to nan loss (near zero weights)
        # [0, 1] to [-max_sigma, max_sigma]
        alpha = alpha * 2 - 1
        alpha = self.max_sigma * alpha
        # alpha = torch.exp(alpha)
        tmp_alpha = torch.nn.functional.pad(alpha, pad=self.pad_vec, mode='replicate')

        factor_alpha = tmp_alpha[:, :, self.field_of_view_x, self.field_of_view_y]

        weights = self.linear_weight(factor_alpha, self.dis_x, self.dis_y)

        # normalize
        weights = torch.reshape(weights, (B, C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = weights.transpose(3, 4)
        weights = torch.reshape(weights, (B, C, out_sz[0], out_sz[1], support_sz*support_sz))
        weights_patch_sum = weights.sum(axis=4, keepdim=True)
        weights = weights / weights_patch_sum

        tmp_input = torch.nn.functional.pad(input, pad=self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[:, :, self.field_of_view_x, self.field_of_view_y] # [B, C, outH*sz, outW*sz]
        neighbors = torch.reshape(neighbors, (B, C, out_sz[0], support_sz, out_sz[1], support_sz))
        neighbors = neighbors.transpose(3, 4)
        neighbors = torch.reshape(neighbors, (B, C, out_sz[0], out_sz[1], support_sz*support_sz))

        output = neighbors * weights
        output = output.sum(4)
        return output
    
