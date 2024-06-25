import os
import sys
from multiprocessing import Pool
import numpy as np
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
# RuntimeWarning: invalid value encountered in true_divide
#   weights = weights / weights_patch_sum

sys.path.insert(0, "./")  # run under the current directory
from common.utils import PSNR, mPSNR
from common.option import TestOptions
from resize_right.resize_right2d_numpy import SteeringGaussianWarp2dNumpy, NearestWarp2dNumpy, AmplifiedLinearWarp2dNumpy
from resample.eval_lut_sr import FourSimplexInterpFaster

mode_pad_dict = {
    "s": 1,
    "d": 2,
    "y": 2,
    "c": 3,
    "t": 3, # X in the paper
}

class eltr:
    def __init__(self, opt, lutDict):
        self.opt = opt
        self.lutDict = lutDict
        self.norm = 255
        self.outC = 3  # hyper channel
        self.modes = opt.modes
        self.modes2 = opt.modes2
        self.stages = opt.stages
        self.border = 4 # shrave borders
        if opt.linear:
            self.resizer = AmplifiedLinearWarp2dNumpy()
            self.outC = 1  # hyper channel
        else:
            self.resizer = SteeringGaussianWarp2dNumpy(
                support_sz=opt.suppSize, max_sigma=opt.maxSigma)
        self.nn_warper = NearestWarp2dNumpy()

    def run(self, dataset, scale_p, num_worker=24):
        folder = os.path.join(self.opt.testDir, dataset, "HR")
        files = os.listdir(folder)
        files = [f for f in files if "png" in f]
        files.sort()

        result_path = os.path.join(
            opt.resultRoot,
            opt.expDir.split("/")[-1],
            dataset,
            scale_p,
        )
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        self.result_path = result_path
        self.dataset = dataset
        self.files = files
        self.scale_p = scale_p
     
        psnr_ssim_s = []
        for i in range(len(self.files)):
            psnr_ssim_s.append(self._worker(i))
        return psnr_ssim_s

    def _worker(self, i):
        # Load LR image
        img_lr = np.array(
            Image.open(
                os.path.join(
                    self.opt.testDir,
                    dataset,
                    self.scale_p,
                    self.files[i],
                )
            )
        ).astype(np.float32)
        if len(img_lr.shape) == 2:
            img_lr = np.expand_dims(img_lr, axis=2)
            img_lr = np.concatenate([img_lr, img_lr, img_lr], axis=2)
        img_input = img_lr.copy()

        # load matrix
        matrix = torch.load(os.path.join(self.opt.testDir, dataset, self.scale_p, self.files[i].replace("png", "pth"))).numpy()
        
        # Load GT image
        img_gt = np.array(
            Image.open(os.path.join(self.opt.testDir,
                       dataset, "HR", self.files[i]))
        )

        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate([img_gt, img_gt, img_gt], axis=2)

        modes, modes2, stages = self.modes, self.modes2, self.stages

        # skip the first stage
        if True:
            for s in range(self.stages - 1):
                pred = 0
                stage = s + 1
                for mode in self.modes:
                    key = "s{}_{}r0".format(str(stage), mode)
                    weight = self.lutDict[key]
                    pad = mode_pad_dict[mode]
                    for r in [0, 1, 2, 3]:
                        img_lr_rot = np.rot90(img_lr, r)
                        h, w, _ = img_lr_rot.shape
                        img_in = np.pad(
                            img_lr_rot, ((0, pad), (0, pad), (0, 0)), mode="edge"
                        ).transpose((2, 0, 1))
                        pred += FourSimplexInterpFaster(
                            weight,
                            img_in,
                            h,
                            w,
                            self.opt.interval,
                            4 - r,
                            upscale=1,
                            mode=mode,
                            oC=1,
                        )
                if (s + 1) == (self.stages - 1):
                    avg_factor, bias, norm = len(self.modes), 0, 1
                else:
                    avg_factor, bias, norm = (
                        len(self.modes) * 4,
                        self.norm // 2,
                        float(self.norm),
                    )
                img_lr = (
                    np.round(np.clip((pred / avg_factor) + bias, 0, self.norm))
                    .astype(np.float32)
                    .transpose((1, 2, 0))
                )  # feat [C, H, W]

        pred = 0
        for mode in self.modes2:
            pad = mode_pad_dict[mode]
            for r in [0, 2]:
                key = "s{}_{}r0".format(str(stages), mode)
                weight = self.lutDict[key]
                img_lr_rot = np.rot90(img_lr, r)
                h, w, _ = img_lr_rot.shape
                img_in = np.pad(
                    img_lr_rot, ((0, pad), (0, pad), (0, 0)), mode="edge"
                ).transpose((2, 0, 1))
                pred += FourSimplexInterpFaster(
                    weight,
                    img_in,
                    h,
                    w,
                    self.opt.interval,
                    4 - r,
                    upscale=1,
                    mode=mode,
                    oC=self.outC,
                )
            for r in [1, 3]:
                key = "s{}_{}r1".format(str(stages), mode)
                weight = self.lutDict[key]
                img_lr_rot = np.rot90(img_lr, r)
                h, w, _ = img_lr_rot.shape
                img_in = np.pad(
                    img_lr_rot, ((0, pad), (0, pad), (0, 0)), mode="edge"
                ).transpose((2, 0, 1))
                pred += FourSimplexInterpFaster(
                    weight,
                    img_in,
                    h,
                    w,
                    self.opt.interval,
                    4 - r,
                    upscale=1,
                    mode=mode,
                    oC=self.outC,
                )

        avg_factor, bias, norm = len(
            self.modes2) * 4, self.norm // 2, float(self.norm)
        img_hyper = (
            np.round(np.clip((pred / avg_factor) + bias, 0, self.norm)).astype(
                np.float32
            )
            / norm
        )  # [C, H, W]

        img_lr = img_lr.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
        img_gt = img_gt.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
        self.resizer.set_shape(img_lr.shape, matrix, img_gt.shape)
        
        all_white = np.array(np.zeros_like(img_lr))
        # shrave borders
        border = self.border
        h, w = all_white.shape[-2:]
        all_white[:,border:h-border,border:w-border] = 255

        self.nn_warper.set_shape(img_lr.shape, matrix, img_gt.shape)
        mask_output = self.nn_warper.warp(all_white)

        if opt.linear:
            img_out = self.resizer.warp(img_lr, img_hyper)
        else:
            C, _, _ = img_hyper.shape
            hyper1_c_idx = list(range(0, C, 3))
            hyper2_c_idx = list(range(1, C + 1, 3))
            hyper3_c_idx = list(range(2, C + 2, 3))

            img_out = self.resizer.warp(
                img_lr,
                img_hyper[hyper1_c_idx, :, :],
                img_hyper[hyper2_c_idx, :, :],
                img_hyper[hyper3_c_idx, :, :],
            )
        img_out = np.clip(np.round(img_out).transpose((1, 2, 0)), 0, self.norm).astype(
            np.uint8
        )
        # make mask


        img_gt = img_gt.transpose((1, 2, 0)) # [C, H, W] to [H, W, C]
        gt_tensor = torch.Tensor(img_gt)
        mask_output = mask_output.transpose((1, 2, 0)) # [C, H, W] to [H, W, C]
        mask_tensor = torch.Tensor(np.array(mask_output==255))
        
        # print(img_gt.shape, img_out.shape, mask_output.shape)

        mpsnr = mPSNR(torch.Tensor(img_out), gt_tensor, mask_tensor)


        # save img_lr (feat)
        img_lr = np.clip(np.round(img_lr).transpose((1, 2, 0)), 0, self.norm).astype(
            np.uint8
        )
        Image.fromarray(img_lr).save(
            os.path.join(
                self.result_path,
                "{}_lr.png".format(
                    self.files[i].split("/")[-1][:-4],
                ),
            )
        )
        Image.fromarray(np.array((mask_output==255)*255).astype(np.uint8)).save(
            os.path.join(
                self.result_path,
                "{}_mask.png".format(
                    self.files[i].split("/")[-1][:-4],
                ),
            )
        )
        
        
        # non valid pixes leave as white
        all_white_gt = (np.ones_like(np.array(img_gt))*255).astype(np.uint8)
        img_out = img_out * np.array(mask_output==255) + np.array(mask_output!=255) * all_white_gt
        img_gt = np.array(img_gt) * np.array(mask_output==255) + np.array(mask_output!=255) * all_white_gt
        
        Image.fromarray(img_out).save(
            os.path.join(
                self.result_path,
                "{}_{}.png".format(
                    self.files[i].split("/")[-1][:-4],
                    self.opt.lutName,
                ),
            )
        )

        Image.fromarray(img_gt).save(
            os.path.join(
                self.result_path,
                "{}_gt.png".format(
                    self.files[i].split("/")[-1][:-4],
                ),
            )
        )

        # Save hyper heatmaps
        if False:
            img_hyper = img_hyper.transpose((1, 2, 0))
            H, W, C = img_hyper.shape
            for c in range(C):
                a = img_hyper[:, :, c]
                # manual heatmap
                heat = np.zeros([H, W, 3])
                heat[:, :, c // 3] = np.clip(np.round(a * 255.0), 0, 255)
                Image.fromarray(heat.astype(np.uint8)).save(
                    os.path.join(
                        self.result_path,
                        "{}_{}_heat_{}.png".format(
                            self.files[i].split("_")[-1],
                            self.opt.lutName,
                            str(c),
                        ),
                    )
                )
        
        return [mpsnr]


if __name__ == "__main__":
    opt = TestOptions().parse()

    # Load LUT
    lutDict = dict()
    for s in range(opt.stages):
        stage = s + 1
        cur_modes = opt.modes
        rots = ["0"]
        oC = 1
        if stage == opt.stages:  # hyper stage
            cur_modes = opt.modes2
            rots = ["0", "1"]
            oC = 3
            if opt.linear:
                oC = 1
        for mode in cur_modes:
            for r in rots:
                key = "s{}_{}r{}".format(str(s + 1), mode, r)
                lutPath = os.path.join(
                    opt.expDir,
                    "{}_s{}_{}r{}.npy".format(
                        opt.lutName, str(stage), mode, r
                    ),
                )
                lutDict[key] = (
                    np.array(np.load(lutPath)).astype(
                        np.float32).reshape(-1, oC)
                )

    etr = eltr(opt, lutDict)

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
            psnr_ssim_s = etr.run(dataset, scale_p, 8)
            avg_psnr = np.mean(np.asarray(psnr_ssim_s)[:, 0])
            metric_list.append("{:.2f}".format(avg_psnr))
            # print(scale_p, avg_psnr)
        print("\t".join(metric_list))
