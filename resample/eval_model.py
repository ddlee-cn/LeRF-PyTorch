import os
import sys
import numpy as np
import torch
from multiprocessing import Pool
from PIL import Image

import model

sys.path.insert(0, "./")  # run under the project directory
from common.option import TestOptions
from resize_right.resize_right2d_torch import SteeringGaussianResize2dTorch, SteeringGaussianWarp2dTorch, NearestWarp2dTorch
from common.utils import PSNR, mPSNR, cal_ssim, logger_info, _rgb2ycbcr

mode_pad_dict = {
    "s": 1,
    "d": 2,
    "y": 2,
    "c": 3,
    "t": 3, # X in the paper
}

nn_warper = NearestWarp2dTorch(device=torch.device("cuda"))

def mulut_predict(model_G, x, stage=1, phase="train", opt=None):
    with torch.no_grad():
        if opt.inC == 1:
            result = []
            for i in range(x.shape[1]):
                result.append(model_G.predict(x[:, i:i+1, :, :], stage=stage))
            return torch.cat(result, dim=1)
        else:
            return model_G.predict(x, stage=stage)
    return x


class eltr:
    def __init__(self, opt, model_G):
        self.opt = opt
        self.model_G = model_G
        self.norm = 255
        self.outC = 3  # hyper-param channel
        self.modes = opt.modes
        self.modes2 = opt.modes2
        self.resizer = SteeringGaussianResize2dTorch(support_sz=opt.suppSize, device=torch.device("cuda"), max_sigma=opt.maxSigma)
        self.warper = SteeringGaussianWarp2dTorch(support_sz=opt.suppSize, device=torch.device("cuda"), max_sigma=opt.maxSigma)

    def run(self, dataset, scale_h, scale_w):
        folder = os.path.join(self.opt.testDir, dataset, "HR")
        files = os.listdir(folder)
        files = [f for f in files if "png" in f]
        files.sort()

        result_path = os.path.join(
            opt.resultRoot,
            opt.expDir.split("/")[-1],
            "X{:.2f}_{:.2f}".format(scale_h, scale_w),
            dataset,
        )
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        self.result_path = result_path
        self.dataset = dataset
        self.files = files
        self.scale_h = scale_h
        self.scale_w = scale_w

        psnr_ssim_s = []
        for i in range(len(self.files)):
            psnr, ssim = self._worker(i)
            psnr_ssim_s.append([psnr, ssim])
        return psnr_ssim_s
    
    def run_warp(self, scale, dataset):
        folder = os.path.join(self.opt.testDir, dataset, "HR")
        files = os.listdir(folder)
        files.sort()

        result_path = os.path.join(opt.resultRoot, dataset, "warp_{}".format(str(scale)))
        if not os.path.isdir(result_path):
            os.makedirs(result_path)
            
        self.result_path = result_path
        self.dataset = dataset
        self.files = files
        self.scale = scale
        
        psnr_ssim_s = []
        for i in range(len(self.files)):
            psnr, ssim = self._worker_warp(i)
            psnr_ssim_s.append([psnr, ssim])
        return psnr_ssim_s

    def _worker(self, i):
        scale_h, scale_w = self.scale_h, self.scale_w
        # Load LR image
        img_lr = np.array(
            Image.open(
                os.path.join(
                    self.opt.testDir,
                    dataset,
                    "LR_bicubic/rrLR_X{:.2f}_{:.2f}".format(scale_h, scale_w),
                    self.files[i],
                )
            )
        ).astype(np.float32)
        if len(img_lr.shape) == 2:
            img_lr = np.expand_dims(img_lr, axis=2)
            img_lr = np.concatenate([img_lr, img_lr, img_lr], axis=2)

        img_lr = (
            torch.Tensor(img_lr)[None, :].permute(0, 3, 1, 2)
            / 255.0
        ).cuda()
        # Load GT image
        img_gt = np.array(
            Image.open(os.path.join(self.opt.testDir, dataset, "HR", self.files[i]))
        )

        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate([img_gt, img_gt, img_gt], axis=2)
            
        if opt.twoStage:
            feat_im = mulut_predict(model_G, img_lr, 1, 'valid', opt)
            hyper_in = feat_im / float(opt.norm)
        else:
            feat_im = torch.round(img_lr * opt.norm)
            hyper_in = img_lr
            
        pred_hyper = mulut_predict(model_G, hyper_in, 2, 'valid', opt)

        # set shape per image
        post = 1
        if "PreUpsample" in opt.testDir:
            post = 2

        input_scale_h, input_scale_w = self.scale_h / post, self.scale_w / post
        self.resizer.set_shape(img_lr.shape, [input_scale_h, input_scale_w])
        
        if opt.inC == 1:
            _, C, _, _ = pred_hyper.shape
            hyper1_c_idx = list(range(0, C, 3))
            hyper2_c_idx = list(range(1, C+1, 3))
            hyper3_c_idx = list(range(2, C+2, 3))
            pred  = self.resizer.resize(feat_im, pred_hyper[:, hyper1_c_idx, :, :], pred_hyper[:, hyper2_c_idx, :, :], pred_hyper[:, hyper3_c_idx, :, :])
        else:
            pred  = self.resizer.resize(feat_im, pred_hyper[:, :1*opt.featC, :, :], pred_hyper[:, 1*opt.featC:2*opt.featC, :, :], pred_hyper[:, 2*opt.featC:, :, :])
        
        
        # skipping
        if input_scale_h == 1 and input_scale_w == 1:
            pred = torch.round(img_lr * opt.norm)
            
        
        pred = np.transpose(np.squeeze(
            pred.data.cpu().numpy(), 0), [1, 2, 0])

        torch.cuda.empty_cache()
        
        img_out = np.clip(np.round(pred), 0, self.norm).astype(
            np.uint8
        )

        # Save to file
        Image.fromarray(img_out).save(
            os.path.join(
                self.result_path, "{}.png".format(self.files[i].split("/")[-1][:-4])
            )
        )

        if img_gt.shape != img_out.shape:
            predH, predW, _ = img_out.shape
            img_gt = img_gt[:predH, :predW, :]
        y_gt, y_out = _rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(img_out)[:, :, 0]
        psnr = PSNR(y_gt, y_out, max(int(self.scale_h), int(self.scale_w)))
        ssim = cal_ssim(y_gt, y_out)
        
        return [psnr, ssim]

    def _worker_warp(self, i):
        scale = self.scale
        # Load LR image
        img_lr = np.array(
            Image.open(
                os.path.join(
                    self.opt.testDir,
                    dataset,
                    scale,
                    self.files[i],
                )
            )
        ).astype(np.float32)
        if len(img_lr.shape) == 2:
            img_lr = np.expand_dims(img_lr, axis=2)
            img_lr = np.concatenate([img_lr, img_lr, img_lr], axis=2)

        img_lr = (
            torch.Tensor(img_lr)[None, :].permute(0, 3, 1, 2)
            / 255.0
        ).cuda()
        # Load GT image
        img_gt = np.array(
            Image.open(os.path.join(self.opt.testDir, dataset, "HR", self.files[i]))
        )


        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate([img_gt, img_gt, img_gt], axis=2)
            
        lb_tensor = torch.Tensor(np.expand_dims(
            np.transpose(img_gt, [2, 0, 1]), axis=0)).cuda()    
        
        # load matrix
        matrix = torch.load(os.path.join(self.opt.testDir, dataset, scale, self.files[i].replace("png", "pth"))).numpy()
        
        m = torch.Tensor(matrix).double().cuda()
        if "PreUpsample" in opt.testDir:
            post = torch.Tensor([
                [0.5, 0, -0.25],
                [0, 0.5, -0.25],
                [0, 0, 1],
            ])
            m = torch.matmul(m, post.double().cuda())

        if opt.twoStage:
            feat_im = mulut_predict(model_G, img_lr, 1, 'valid', opt)
            hyper_in = feat_im / float(opt.norm)
        else:
            feat_im = torch.round(img_lr * opt.norm)
            hyper_in = img_lr
            
        pred_hyper = mulut_predict(model_G, hyper_in, 2, 'valid', opt)

        # get mask
        all_white = torch.zeros_like(img_lr)
        # shrave borders
        border = 4
        h, w = all_white.shape[-2:]
        all_white[:,:,border:h-border,border:w-border] = 255

        nn_warper.set_shape(img_lr.shape, m, lb_tensor.shape)
        mask_output = nn_warper.warp(all_white).bool()
        # mask_tensor = torch.Tensor(mask_output).bool()

        # set shape per image
        self.warper.set_shape(img_lr.shape, m, lb_tensor.shape)
        
        if opt.inC == 1:
            _, C, _, _ = pred_hyper.shape
            hyper1_c_idx = list(range(0, C, 3))
            hyper2_c_idx = list(range(1, C+1, 3))
            hyper3_c_idx = list(range(2, C+2, 3))
            pred  = self.warper.warp(feat_im, pred_hyper[:, hyper1_c_idx, :, :], pred_hyper[:, hyper2_c_idx, :, :], pred_hyper[:, hyper3_c_idx, :, :])
        else:
            pred  = self.warper.warp(feat_im, pred_hyper[:, :1*opt.featC, :, :], pred_hyper[:, 1*opt.featC:2*opt.featC, :, :], pred_hyper[:, 2*opt.featC:, :, :])
        
        
        pred[pred.isnan()] = 0
        
            
        
        pred = torch.round(pred.clip(0, 255))

        # print(pred.shape, lb_tensor.shape, mask_output.shape)
        psnr = mPSNR(pred, lb_tensor, mask_output, 255)
        
        pred = np.transpose(np.squeeze(
            pred.data.cpu().numpy(), 0), [1, 2, 0])
        
        torch.cuda.empty_cache()

        img_out = np.clip(np.round(pred), 0, self.norm).astype(
            np.uint8
        )

        # Save to file
        Image.fromarray(img_out).save(
            os.path.join(
                self.result_path, "{}.png".format(self.files[i].split("/")[-1][:-4])
            )
        )
        
        return [psnr.item(), 0]

if __name__ == "__main__":
    opt = TestOptions().parse()

    modes = [i for i in opt.modes]

    model = getattr(model, opt.model)

    model_G = model(opt, inC=opt.inC, outC=opt.outC).cuda()

    # Load saved network params
    lm = torch.load(os.path.join(opt.expDir, "Model_{:06d}.pth".format(opt.loadIter)))

    model_G.load_state_dict(lm.state_dict(), strict=True)
    model_G.eval()

    etr = eltr(opt, model_G)
    

        
    if "warp" in opt.resultRoot:
        all_datasets = ["Set5"]
        # all_datasets = ["Set5", "Set14", "B100", "Urban100", "DIV2KValid"]
        all_scales = [
            "isc", "osc"
        ]

        scale_head = ["Scale".ljust(15, " ")]
        for scale_p in all_scales:
            scale_head.append("{}\t".format(scale_p))
        print("\t".join(scale_head))

        for dataset in all_datasets:
            metric_list = [dataset.ljust(15, " ")]
            for scale_p in all_scales:
                psnr_ssim_s = etr.run_warp(scale_p, dataset)
                avg_psnr = np.mean(np.asarray(psnr_ssim_s)[:, 0])
                metric_list.append("{:.2f}".format(avg_psnr))
                # print(scale_p, avg_psnr)
            print("\t".join(metric_list))
    else:

        # all_datasets = ["Set5", "Set14", "B100", "Urban100", "Manga109", "div2k"]
        # all_scales = [
        #     [1.5, 1.5],
        #     [1.5, 2],
        #     [2, 2],
        #     [2, 2.4],
        #     [2, 3],
        #     [3, 3],
        #     [3, 4],
        #     [4, 4],
        # ]
        all_datasets = ["Set5"]
        all_scales = [[2, 2], [3, 3], [4, 4]]

        scale_head = ["Scale".ljust(15, " ")]
        for scale_p in all_scales:
            scale_h, scale_w = scale_p
            scale_head.append("{:.1f}x{:.1f}\t".format(scale_h, scale_w))
        print("\t".join(scale_head))

        for dataset in all_datasets:
            metric_list = [dataset.ljust(15, " ")]
            if dataset == "div2k":
                all_scales = [[2, 2], [3, 3], [4, 4]]
            for scale_p in all_scales:
                scale_h, scale_w = scale_p
                psnr_ssim_s = etr.run(dataset, scale_h, scale_w)
                avg_psnr, avg_ssim = np.mean(np.asarray(psnr_ssim_s)[:, 0]), np.mean(
                    np.asarray(psnr_ssim_s)[:, 1]
                )
                metric_list.append("{:.2f}/{:.4f}".format(avg_psnr, avg_ssim))
            print("\t".join(metric_list))

