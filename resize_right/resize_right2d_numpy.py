from math import ceil
import numpy as np

import sys
sys.path.insert(0, "./")

import resize_right.interp_methods as interp_methods


class Resize2dNumpy(object):
    def __init__(self, support_sz=4, device="CPU", pad_mode="constant"):
        self.eps = np.finfo(np.float32).eps
        self.device = device
        self.support_sz = support_sz
        self.pad_mode = pad_mode
        self.antialias = False

    def set_shape(self, in_shape, scale_factors=None, out_shape=None):
        # fixed in_shape, scale_factors, and out_shape
        # in_shape [C, H, W]
        self.in_shape = in_shape
        self.set_scale_and_out_sz(in_shape, scale_factors, out_shape)
        self.get_distance(self.in_shape, self.out_shape, self.scale_factors)

    def set_scale_and_out_sz(self, in_shape, scale_factors, out_shape):
        if out_shape is not None:
            out_shape = list(out_shape) + list(in_shape[len(out_shape) :])
            if scale_factors is None:
                scale_factors = [
                    out_sz / in_sz for out_sz, in_sz in zip(out_shape, in_shape)
                ]
        if scale_factors is not None:
            scale_factors = (
                scale_factors
                if isinstance(scale_factors, (list, tuple))
                else [scale_factors, scale_factors]
            )
            scale_factors = [1] * (len(in_shape) - len(scale_factors)) + list(
                scale_factors
            )
            if out_shape is None:
                out_shape = [
                    ceil(scale_factor * in_sz)
                    for scale_factor, in_sz in zip(scale_factors, in_shape)
                ]
        self.scale_factors = [float(s) for s in scale_factors]
        self.out_shape = out_shape
        self.in_sz = [in_shape[1], in_shape[2]]
        self.out_sz = [out_shape[1], out_shape[2]]

        # apply anti-aliasing for downsampling
        if self.scale_factors[0] < 1.0 or self.scale_factors[1] < 1.0:
            self.antialias = True
            self.min_scale_factor = min([self.scale_factors[1], self.scale_factors[0]])
            self.support_sz = ceil(self.support_sz / self.min_scale_factor)

    def get_projected_grid2d(self, scale_factor, in_shape, out_shape):
        # skip B and C dims for scale_factor, in_shape, and out_shape
        in_sz = self.in_sz
        out_sz = self.out_sz
        grid_sz_h = out_sz[0]
        grid_sz_w = out_sz[1]
        x = np.arange(grid_sz_h)
        y = np.arange(grid_sz_w)

        x_r = np.repeat(x, self.support_sz)
        y_r = np.repeat(y, self.support_sz)

        X, Y = np.meshgrid(x_r, y_r, indexing="ij")
        grid_x = (
            X / float(scale_factor[1])
            + (in_sz[0] - 1) / 2
            - (out_sz[0] - 1) / (2 * float(scale_factor[1]))
        )
        grid_y = (
            Y / float(scale_factor[2])
            + (in_sz[1] - 1) / 2
            - (out_sz[1] - 1) / (2 * float(scale_factor[2]))
        )
        return grid_x, grid_y

    def get_field_of_view2d(self, projected_grid_x, projected_grid_y, out_shape):
        out_sz = self.out_sz
        cur_support_sz = self.support_sz
        left_boundaries_x = np.int_(
            np.ceil((projected_grid_x - cur_support_sz / 2 - self.eps))
        )
        left_boundaries_y = np.int_(
            np.ceil((projected_grid_y - cur_support_sz / 2 - self.eps))
        )

        ordinal_numbers_x = np.arange(ceil(cur_support_sz - self.eps))
        ordinal_numbers_y = np.arange(ceil(cur_support_sz - self.eps))

        ord_X, ord_Y = np.meshgrid(ordinal_numbers_x, ordinal_numbers_y)
        ord_X_r = np.tile(ord_X, (out_sz[0], out_sz[1]))
        ord_Y_r = np.tile(ord_Y, (out_sz[0], out_sz[1]))
        return left_boundaries_x + ord_X_r, left_boundaries_y + ord_Y_r

    def calc_pad_sz(self, in_sz, out_sz, field_of_view, projected_grid, scale_factor):
        pad_sz = (-field_of_view[0, 0].item(), field_of_view[-1, -1].item() - in_sz + 1)
        field_of_view += pad_sz[0]
        projected_grid += pad_sz[0]
        return pad_sz, projected_grid, field_of_view

    def get_distance(self, in_shape, out_shape, scale_factors):
        projected_grid_x, projected_grid_y = self.get_projected_grid2d(
            scale_factors, in_shape, out_shape
        )
        field_of_view_x, field_of_view_y = self.get_field_of_view2d(
            projected_grid_x, projected_grid_y, out_shape
        )

        pad_sz_x, projected_grid_x, field_of_view_x = self.calc_pad_sz(
            self.in_sz[0],
            self.out_sz[0],
            field_of_view_x,
            projected_grid_x,
            self.scale_factors[1],
        )
        pad_sz_y, projected_grid_y, field_of_view_y = self.calc_pad_sz(
            self.in_sz[1],
            self.out_sz[1],
            field_of_view_y,
            projected_grid_y,
            self.scale_factors[2],
        )

        self.pad_vec = ((0, 0), pad_sz_x, pad_sz_y)

        dis_x, dis_y = (
            projected_grid_x - field_of_view_x,
            projected_grid_y - field_of_view_y,
        )  # [out_h*sz, out_w*sz]

        self.field_of_view_x, self.field_of_view_y = field_of_view_x, field_of_view_y
        self.dis_x = np.concatenate(
            [np.expand_dims(dis_x, 0)] * in_shape[0], axis=0
        )  # [3, outH*sz, outW*sz]
        self.dis_y = np.concatenate([np.expand_dims(dis_y, 0)] * in_shape[0], axis=0)

class SteeringGaussianResize2dNumpy(Resize2dNumpy):
    def __init__(self, support_sz=4, device="CPU", pad_mode="constant", max_sigma=10):
        super().__init__(support_sz, device, pad_mode)
        self.h = 5.0
        self.support_sz = support_sz
        self.max_sigma = max_sigma
        self.antialias = False

    def sk_weight(self, rho, sigma_x, sigma_y, x, y):
        # [B, C, oH, oW]
        multiplier = 1  # 1 / (2*pi*sigma_x*sigma_y*math.sqrt(1-rho**2) + self.eps)
        e_multiplier = -1 / 2  # * 1/(self.max_sigma) # -1 * (1/(2*(1-rho**2)+self.eps))
        x_nominal = (sigma_x * x) ** 2
        y_nominal = (sigma_y * y) ** 2
        xy_nominal = sigma_x * x * sigma_y * y
        # exp_term = np.clip(e_multiplier * (x_nominal - 2*rho*xy_nominal + y_nominal), -50, 50)
        exp_term = e_multiplier * (x_nominal - 2 * rho * xy_nominal + y_nominal)
        result = multiplier * np.exp(exp_term)
        return result

    def resize(self, input, rho, sigma_x, sigma_y):
        # input: [B, C, H, W]
        C, H, W = input.shape
        out_sz = self.out_sz
        support_sz = self.support_sz

        rho = rho * 2 - 1  # rho: [-1, 1]
        sigma_x = sigma_x * self.max_sigma  # sigma: [0, max_sigma]
        sigma_y = sigma_y * self.max_sigma

        tmp_rho = np.pad(rho, self.pad_vec, mode="edge")
        tmp_sigma_x = np.pad(sigma_x, self.pad_vec, mode="edge")
        tmp_sigma_y = np.pad(sigma_y, self.pad_vec, mode="edge")

        factor_rho = tmp_rho[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        factor_sigma_x = tmp_sigma_x[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        factor_sigma_y = tmp_sigma_y[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]

        if self.antialias:
            weights = self.min_scale_factor * self.sk_weight(
                factor_rho,
                factor_sigma_x,
                factor_sigma_y,
                self.min_scale_factor * self.dis_x,
                self.min_scale_factor * self.dis_y,
            )  # [B, C, outH*sz, outW*sz]
        else:
            weights = self.sk_weight(
                factor_rho, factor_sigma_x, factor_sigma_y, self.dis_x, self.dis_y
            )  # [B, C, outH*sz, outW*sz]

        # normalize
        weights = np.reshape(weights, (C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = np.swapaxes(weights, 2, 3)
        weights = np.reshape(
            weights, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )
        weights_patch_sum = np.sum(weights, axis=3, keepdims=True)
        weights = weights / weights_patch_sum

        tmp_input = np.pad(input, self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], support_sz, out_sz[1], support_sz)
        )
        neighbors = neighbors.swapaxes(2, 3)
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )

        output = neighbors * weights  # [B, C, outH*sz, outW*sz]
        output = np.sum(output, axis=3)  # [B, C, outH, outW]

        return output
    
class AmplifiedLinearResize2dNumpy(Resize2dNumpy):
    def __init__(self, support_sz=2, device="CPU", pad_mode="constant", max_sigma=1):
        super().__init__(support_sz, device, pad_mode)
        self.h = 5.0
        self.support_sz = support_sz
        self.max_sigma = max_sigma
        self.antialias = False
    
    def linear_alpha(self, x, alpha):
        return ((alpha * x + 1) * ((-1 <= x) & (x < 0)) + (1 - alpha * x) *
                ((0 <= x) & (x <= 1)))

    def linear_weight(self, alpha, x, y):
        # weight = linear(x) * linear(y)
        # negative weights are clipped
        weight = np.clip(self.linear_alpha(x, alpha), 0, None) * np.clip(self.linear_alpha(y, alpha), 0, None)
        return weight
    
    def resize(self, input, alpha):
        # input: [B, C, H, W]
        C, H, W = input.shape
        out_sz = self.out_sz
        support_sz = self.support_sz

        alpha = alpha * 2 - 1
        alpha = self.max_sigma * alpha
        # alpha = torch.exp(alpha)
        tmp_alpha = np.pad(alpha, self.pad_vec, mode='edge')

        factor_alpha = tmp_alpha[:, self.field_of_view_x, self.field_of_view_y]

        weights = self.linear_weight(factor_alpha, self.dis_x, self.dis_y)

        # normalize
        weights = np.reshape(weights, (C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = np.swapaxes(weights, 2, 3)
        weights = np.reshape(
            weights, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )
        weights_patch_sum = np.sum(weights, axis=3, keepdims=True)
        weights = weights / weights_patch_sum

        tmp_input = np.pad(input, self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], support_sz, out_sz[1], support_sz)
        )
        neighbors = neighbors.swapaxes(2, 3)
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )

        output = neighbors * weights  # [B, C, outH*sz, outW*sz]
        output = np.sum(output, axis=3)  # [B, C, outH, outW]

        return output
    
class Warp2dNumpy(object):
    def __init__(self, support_sz=4, device="CPU", pad_mode="constant"):
        self.eps = np.finfo(np.float32).eps
        self.device = device
        self.support_sz = support_sz
        self.pad_mode = pad_mode
        self.antialias = False

    def set_shape(self, in_shape, matrix, out_shape):
        # fixed in_shape, scale_factors, and out_shape
        # in_shape [C, H, W]
        self.in_shape = in_shape
        self.matrix = matrix
        self.set_scale_and_out_sz(in_shape, out_shape)
        self.get_distance(self.in_shape, self.out_shape, self.matrix)

    def set_scale_and_out_sz(self, in_shape, out_shape):
        out_shape = list(out_shape) + list(in_shape[len(out_shape) :])
        self.out_shape = out_shape
        self.in_sz = [in_shape[1], in_shape[2]]
        self.out_sz = [out_shape[1], out_shape[2]]
    
    def get_projected_grid2d(self, matrix, in_shape, out_shape):
        # skip B and C dims for scale_factor, in_shape, and out_shape
        in_sz = self.in_sz
        out_sz = self.out_sz
        grid_sz_h = out_sz[0]
        grid_sz_w = out_sz[1]
        x = np.arange(grid_sz_h)
        y = np.arange(grid_sz_w)

        x_r = np.repeat(x, self.support_sz)
        y_r = np.repeat(y, self.support_sz)

        grid = np.meshgrid(x_r, y_r, indexing="ij")
        
        # gridy = make_coord(self.out_sz, self.support_sz)
        ret = np.stack(grid, axis=-1)
        gridy = ret.reshape(-1, ret.shape[-1]).astype(np.float32)
        
        # h -> y, w -> x
        gridy = np.flip(gridy, -1)
        # gridy = make_coord(out_sz, support_sz)
        gridx = np.dot(np.linalg.inv(matrix), np.concatenate([gridy, np.ones([gridy.shape[0], 1])], axis=-1).transpose(1, 0)).transpose(1, 0)   
        
        h, w = self.in_sz
        gridx[:, 0] /= gridx[:, -1]
        gridx[:, 1] /= gridx[:, -1]
        gridx = gridx[:, 0:2]     
        
        # reverse back
        gridx = np.flip(gridx, -1)

        # clipped out of field coordinates
        grid_x = gridx[:, 0].reshape(grid_sz_h*self.support_sz, grid_sz_w*self.support_sz).clip(0, in_sz[0])
        grid_y = gridx[:, 1].reshape(grid_sz_h*self.support_sz, grid_sz_w*self.support_sz).clip(0, in_sz[1])
        
        # print(grid_x, grid_y)
        return grid_x, grid_y

    def get_field_of_view2d(self, projected_grid_x, projected_grid_y, out_shape):
        out_sz = self.out_sz
        cur_support_sz = self.support_sz
        left_boundaries_x = np.int_(
            np.ceil((projected_grid_x - cur_support_sz / 2 - self.eps))
        )
        left_boundaries_y = np.int_(
            np.ceil((projected_grid_y - cur_support_sz / 2 - self.eps))
        )
        
        ordinal_numbers_x = np.arange(ceil(cur_support_sz - self.eps))
        ordinal_numbers_y = np.arange(ceil(cur_support_sz - self.eps))

        ord_X, ord_Y = np.meshgrid(ordinal_numbers_x, ordinal_numbers_y)
        ord_X_r = np.tile(ord_X, (out_sz[0], out_sz[1]))
        ord_Y_r = np.tile(ord_Y, (out_sz[0], out_sz[1]))
        
        return left_boundaries_x + ord_X_r, left_boundaries_y + ord_Y_r

    def calc_pad_sz(self, in_sz, out_sz, field_of_view, projected_grid, scale_factor=None):
        # prevent negative pad_width
        pad_sz = (max(-field_of_view[0, 0].item(), 0), max(field_of_view[-1, -1].item() - in_sz + 1, 0))
        field_of_view += pad_sz[0]
        projected_grid += pad_sz[0]
        
        return pad_sz, projected_grid, field_of_view

    def get_distance(self, in_shape, out_shape, matrix):
        projected_grid_x, projected_grid_y = self.get_projected_grid2d(
            matrix, in_shape, out_shape
        )
        field_of_view_x, field_of_view_y = self.get_field_of_view2d(
            projected_grid_x, projected_grid_y, out_shape
        )

        pad_sz_x, projected_grid_x, field_of_view_x = self.calc_pad_sz(
            self.in_sz[0],
            self.out_sz[0],
            field_of_view_x,
            projected_grid_x,
        )
        pad_sz_y, projected_grid_y, field_of_view_y = self.calc_pad_sz(
            self.in_sz[1],
            self.out_sz[1],
            field_of_view_y,
            projected_grid_y,
        )

        self.pad_vec = ((0, 0), pad_sz_x, pad_sz_y)
        self.projected_grid_x, self.projected_grid_y = projected_grid_x, projected_grid_y
        self.field_of_view_x, self.field_of_view_y = field_of_view_x, field_of_view_y

        # clip out of field pixels
        self.field_of_view_x = field_of_view_x.clip(0, self.in_sz[0]-1)
        self.field_of_view_y = field_of_view_y.clip(0, self.in_sz[1]-1)

        dis_x, dis_y = (
            self.projected_grid_x - self.field_of_view_x,
            self.projected_grid_y - self.field_of_view_y,
        )  # [out_h*sz, out_w*sz]

        # [3, outH*sz, outW*sz]
        self.dis_x = np.concatenate([np.expand_dims(dis_x, 0)] * in_shape[0], axis=0)  
        self.dis_y = np.concatenate([np.expand_dims(dis_y, 0)] * in_shape[0], axis=0)

    def warp(self, input):
        # input: [B, C, H, W]
        C, H, W = input.shape
        out_sz = self.out_sz
        support_sz = self.support_sz

        if self.antialias:
            weights = self.min_scale_factor * self.weight(
                self.min_scale_factor * self.dis_x,
                self.min_scale_factor * self.dis_y,
            )  # [B, C, outH*sz, outW*sz]
        else:
            weights = self.weight(self.dis_x, self.dis_y)  # [B, C, outH*sz, outW*sz]

        # normalize
        weights = np.reshape(weights, (C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = np.swapaxes(weights, 2, 3)
        weights = np.reshape(
            weights, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )
        weights_patch_sum = np.sum(weights, axis=3, keepdims=True)
        # divide zero warning for warping mask, ignore
        weights = weights / weights_patch_sum

        tmp_input = np.pad(input, self.pad_vec, mode=self.pad_mode)
        # print(input.shape)
        neighbors = tmp_input[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], support_sz, out_sz[1], support_sz)
        )
        neighbors = neighbors.swapaxes(2, 3)
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )

        output = neighbors * weights  # [B, C, outH*sz, outW*sz]
        output = np.sum(output, axis=3)  # [B, C, outH, outW]

        return output
    
class BicubicWarp2dNumpy(Warp2dNumpy):
    def __init__(self, support_sz=4, device="CPU", pad_mode="constant"):
        super().__init__(support_sz, device, pad_mode)
        self.support_sz = support_sz
        self.antialias = False
        
    def weight(self, x, y):
        return interp_methods.cubic2d(x, y)
    
class NearestWarp2dNumpy(Warp2dNumpy):
    def __init__(self, support_sz=1, device="CPU", pad_mode="constant"):
        super().__init__(support_sz, device, pad_mode)
        self.support_sz = support_sz
        self.antialias = False
        
    def weight(self, x, y):
        return interp_methods.box2d(x, y)

class BilinearWarp2dNumpy(Warp2dNumpy):
    def __init__(self, support_sz=2, device="CPU", pad_mode="constant"):
        super().__init__(support_sz, device, pad_mode)
        self.support_sz = support_sz
        self.antialias = False
        
    def weight(self, x, y):
        return interp_methods.linear2d(x, y)

class Lanczos2Warp2dNumpy(Warp2dNumpy):
    def __init__(self, support_sz=4, device="CPU", pad_mode="constant"):
        super().__init__(support_sz, device, pad_mode)
        self.support_sz = support_sz
        self.antialias = False
        
    def weight(self, x, y):
        return interp_methods.lanczos2d(x, y)

class Lanczos3Warp2dNumpy(Warp2dNumpy):
    def __init__(self, support_sz=6, device="CPU", pad_mode="constant"):
        super().__init__(support_sz, device, pad_mode)
        self.support_sz = support_sz
        self.antialias = False
        
    def weight(self, x, y):
        return interp_methods.lanczos3d(x, y)

class SteeringGaussianWarp2dNumpy(Warp2dNumpy):
    def __init__(self, support_sz=4, device="CPU", pad_mode="constant", max_sigma=10):
        super().__init__(support_sz, device, pad_mode)
        self.h = 5.0
        self.support_sz = support_sz
        self.max_sigma = max_sigma
        self.antialias = False

    def sk_weight(self, rho, sigma_x, sigma_y, x, y):
        # [B, C, oH, oW]
        multiplier = 1  # 1 / (2*pi*sigma_x*sigma_y*math.sqrt(1-rho**2) + self.eps)
        e_multiplier = -1 / 2  # * 1/(self.max_sigma) # -1 * (1/(2*(1-rho**2)+self.eps))
        x_nominal = (sigma_x * x) ** 2
        y_nominal = (sigma_y * y) ** 2
        xy_nominal = sigma_x * x * sigma_y * y
        # exp_term = np.clip(e_multiplier * (x_nominal - 2*rho*xy_nominal + y_nominal), -50, 50)
        exp_term = e_multiplier * (x_nominal - 2 * rho * xy_nominal + y_nominal)
        result = multiplier * np.exp(exp_term)
        return result
    
    def warp(self, input, rho, sigma_x, sigma_y):
        # input: [B, C, H, W]
        C, H, W = input.shape
        out_sz = self.out_sz
        support_sz = self.support_sz

        rho = rho * 2 - 1  # rho: [-1, 1]
        sigma_x = sigma_x * self.max_sigma  # sigma: [0, max_sigma]
        sigma_y = sigma_y * self.max_sigma

        tmp_rho = np.pad(rho, self.pad_vec, mode="edge")
        tmp_sigma_x = np.pad(sigma_x, self.pad_vec, mode="edge")
        tmp_sigma_y = np.pad(sigma_y, self.pad_vec, mode="edge")

        factor_rho = tmp_rho[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        factor_sigma_x = tmp_sigma_x[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        factor_sigma_y = tmp_sigma_y[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]

        if self.antialias:
            weights = self.min_scale_factor * self.sk_weight(
                factor_rho,
                factor_sigma_x,
                factor_sigma_y,
                self.min_scale_factor * self.dis_x,
                self.min_scale_factor * self.dis_y,
            )  # [B, C, outH*sz, outW*sz]
        else:
            weights = self.sk_weight(
                factor_rho, factor_sigma_x, factor_sigma_y, self.dis_x, self.dis_y
            )  # [B, C, outH*sz, outW*sz]

        # normalize
        weights = np.reshape(weights, (C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = np.swapaxes(weights, 2, 3)
        weights = np.reshape(
            weights, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )
        weights_patch_sum = np.sum(weights, axis=3, keepdims=True)
        weights = weights / weights_patch_sum

        tmp_input = np.pad(input, self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], support_sz, out_sz[1], support_sz)
        )
        neighbors = neighbors.swapaxes(2, 3)
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )

        output = neighbors * weights  # [B, C, outH*sz, outW*sz]
        output = np.sum(output, axis=3)  # [B, C, outH, outW]

        return output
    
class AmplifiedLinearWarp2dNumpy(Warp2dNumpy):
    def __init__(self, support_sz=2, device="CPU", pad_mode="constant", max_sigma=1):
        super().__init__(support_sz, device, pad_mode)
        self.h = 5.0
        self.support_sz = support_sz
        self.max_sigma = max_sigma
        self.antialias = False
    
    def linear_alpha(self, x, alpha):
        return ((alpha * x + 1) * ((-1 <= x) & (x < 0)) + (1 - alpha * x) *
                ((0 <= x) & (x <= 1)))

    def linear_weight(self, alpha, x, y):
        # weight = linear(x) * linear(y)
        # negative weights are clipped
        weight = np.clip(self.linear_alpha(x, alpha), 0, None) * np.clip(self.linear_alpha(y, alpha), 0, None)
        return weight
    
    def warp(self, input, alpha):
        # input: [B, C, H, W]
        C, H, W = input.shape
        out_sz = self.out_sz
        support_sz = self.support_sz

        alpha = alpha * 2 - 1
        alpha = self.max_sigma * alpha
        # alpha = torch.exp(alpha)
        tmp_alpha = np.pad(alpha, self.pad_vec, mode='edge')

        factor_alpha = tmp_alpha[:, self.field_of_view_x, self.field_of_view_y]

        weights = self.linear_weight(factor_alpha, self.dis_x, self.dis_y)

        # normalize
        weights = np.reshape(weights, (C, out_sz[0], support_sz, out_sz[1], support_sz))
        weights = np.swapaxes(weights, 2, 3)
        weights = np.reshape(
            weights, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )
        weights_patch_sum = np.sum(weights, axis=3, keepdims=True)
        weights = weights / weights_patch_sum

        tmp_input = np.pad(input, self.pad_vec, mode=self.pad_mode)
        neighbors = tmp_input[
            :, self.field_of_view_x, self.field_of_view_y
        ]  # [B, C, outH*sz, outW*sz]
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], support_sz, out_sz[1], support_sz)
        )
        neighbors = neighbors.swapaxes(2, 3)
        neighbors = np.reshape(
            neighbors, (C, out_sz[0], out_sz[1], support_sz * support_sz)
        )

        output = neighbors * weights  # [B, C, outH*sz, outW*sz]
        output = np.sum(output, axis=3)  # [B, C, outH, outW]

        return output