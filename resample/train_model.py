import math
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
# RuntimeWarning: UserWarning: floor_divide is deprecated

from data import Provider, MultiSRBenchmark, SRBenchmarkW
import model

sys.path.insert(0, "./")  # run under the project directory
import resize_right.interp_methods as interp
from common.option import TrainOptions
from common.utils import PSNR, mPSNR, cal_ssim, logger_info, _rgb2ycbcr
from resize_right.resize_right2d_torch import SteeringGaussianResize2dTorch, SteeringGaussianWarp2dTorch, NearestWarp2dTorch, AmplifiedLinearResize2dTorch, AmplifiedLinearWarp2dTorch

torch.backends.cudnn.benchmark = True

mode_pad_dict = {
    "s": 1,
    "d": 2,
    "y": 2,
    "c": 3,
    "t": 3, # X in the paper
}

def mulut_predict(model_G, x, stage=1, phase="train", opt=None):
    if opt.inC == 1:
        result = []
        for i in range(x.shape[1]):
            result.append(model_G.predict(x[:, i:i+1, :, :], stage=stage))
        return torch.cat(result, dim=1)
    else:
        return model_G.predict(x, stage=stage)
    return x

def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable) with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def SaveCheckpoint(model_G, opt_G, opt, i, best=False):
    str_best = ''
    if best:
        str_best = '_best'

    torch.save(model_G, os.path.join(
            opt.expDir, 'Model_{:06d}{}.pth'.format(i, str_best)))
    # torch.save(opt_G, os.path.join(
        # opt.expDir, 'Opt_{:06d}{}.pth'.format(i, str_best)))
    logger.info("Checkpoint saved {}".format(str(i)))


def valid_steps(model_G, valid, opt, iter):
    scales = [2, 3, 4]
    model_name = opt.expDir.split("/")[-1]
    datasets = valid.datasets

    scale_head = ["Iter {:06d}".format(iter).ljust(15, " ")]
    for scale in scales:
        scale_head.append("X{}\t".format(scale))
    logger.info("\t".join(scale_head))

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            metric_list = [datasets[i].ljust(15, " ")]
            for scale in scales:
                psnrs, ssims = [], []
                files = valid.files[datasets[i]]

                result_path = os.path.join(opt.valoutDir, datasets[i], "X{}".format(str(scale)))
                if not os.path.isdir(result_path):
                    os.makedirs(result_path)

                for j in range(len(files)):
                    key = datasets[i] + '_' + files[j][:-4]

                    lb = valid.ims[key + 'hr']
                    input_im = valid.ims[key + 'X%d' % scale]
                    input_im = input_im.astype(np.float32) / 255.0
                    im = torch.Tensor(np.expand_dims(
                        np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()

                    if opt.twoStage:
                        feat_im = mulut_predict(model_G, im, 1, 'valid', opt)
                        hyper_in = feat_im / float(opt.norm)
                    else:
                        feat_im = torch.round(im * opt.norm)
                        hyper_in = im

                    pred_hyper = mulut_predict(model_G, hyper_in, 2, 'valid', opt)

                    # set shape per image
                    post = 1
                    if "PreUpsample" in opt.valDir:
                        post = 2

                    input_scale = scale / post
                    valid_resizer.set_shape(feat_im.shape, scale_factors=input_scale)
                    if opt.linear:
                        # linear Resmapling Function
                        pred = valid_resizer.resize(feat_im, pred_hyper)
                    else:
                        # Gaussian
                        if opt.inC == 1:
                            _, C, _, _ = pred_hyper.shape
                            hyper1_c_idx = list(range(0, C, 3))
                            hyper2_c_idx = list(range(1, C+1, 3))
                            hyper3_c_idx = list(range(2, C+2, 3))

                            pred = valid_resizer.resize(feat_im, pred_hyper[:, hyper1_c_idx, :, :], pred_hyper[:, hyper2_c_idx, :, :], pred_hyper[:, hyper3_c_idx, :, :])
                        else:
                            pred  = valid_resizer.resize(feat_im, pred_hyper[:, :1*opt.featC, :, :], pred_hyper[:, 1*opt.featC:2*opt.featC, :, :], pred_hyper[:, 2*opt.featC:, :, :])
                        
                    pred = np.transpose(np.squeeze(
                        pred.data.cpu().numpy(), 0), [1, 2, 0])
                    pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                    if lb.shape != pred.shape:
                        predH, predW, _ = pred.shape
                        lb = lb[:predH, :predW, :]
                    left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                    psnrs.append(PSNR(left, right, scale)) # single channel, no scale change
                    ssims.append(cal_ssim(left, right))

                    if iter < 5000 and "div2k" not in datasets[i]: # save input and gt at start
                        input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                        Image.fromarray(input_img).save(
                            os.path.join(result_path, '{}_input.png'.format(key.split('_')[-1])))
                        Image.fromarray(lb.astype(np.uint8)).save(
                            os.path.join(result_path, '{}_gt.png'.format(key.split('_')[-1])))
                        
                    
                    if opt.featC == 3:
                        feat = np.transpose(np.squeeze(
                            feat_im.data.cpu().numpy(), 0), [1, 2, 0])
                        feat = np.round(np.clip(feat, 0, 255)).astype(np.uint8)
                        Image.fromarray(feat).save(
                            os.path.join(result_path, '{}_{}_feat.png'.format(key.split('_')[-1], opt.name)))
                    if "div2k" not in datasets[i]:
                        pred_hyper = np.transpose(np.squeeze(pred_hyper.data.cpu().numpy(), 0), [1, 2, 0])

                        if pred_hyper.shape[-1]==9 and opt.debug:
                            H, W, C = pred_hyper.shape
                            for c in range(C):
                                a = pred_hyper[:, :, c]
                                heat = np.zeros([H, W, 3])
                                heat[:, :, c//3] = np.clip(np.round(a * 255.0), 0, 255)
                                Image.fromarray(heat.astype(np.uint8)).save(os.path.join(result_path, "{}_{}_heat_{}.png".format(key.split('_')[-1], opt.name, str(c))))
                        np.save(os.path.join(result_path, "{}_{}_pred_hyper.npy".format(key.split('_')[-1], opt.name)), pred_hyper)
                        Image.fromarray(pred).save(
                            os.path.join(result_path, '{}_{}_output.png'.format(key.split('_')[-1], opt.name)))
                
                avg_psnr, avg_ssim = np.mean(np.asarray(psnrs)), np.mean(np.asarray(ssims))
                metric_list.append("{:.2f}/{:.4f}".format(avg_psnr, avg_ssim))

                if not opt.lutft:
                    writer.add_scalar('PSNR_X{}/{}'.format(scale, datasets[i]), avg_psnr, iter)
                    writer.add_scalar('SSIM_X{}/{}'.format(scale, datasets[i]), avg_ssim, iter)
                    writer.flush()

            logger.info('\t'.join(metric_list))


def valid_steps_warp(model_G, valid, opt, iter):
    # scales = [2, 3, 4]
    model_name = opt.expDir.split("/")[-1]
    scales = ["isc", "osc"]
    datasets = valid.datasets

    scale_head = ["Iter {:06d}".format(iter).ljust(15, " ")]
    for scale in scales:
        scale_head.append("{}\t".format(scale))
    logger.info("\t".join(scale_head))

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            metric_list = [datasets[i].ljust(15, " ")]
            for scale in scales:
                psnrs = []
                files = valid.files[datasets[i]]

                result_path = os.path.join(opt.valoutDir, datasets[i], "warp_{}".format(str(scale)))
                if not os.path.isdir(result_path):
                    os.makedirs(result_path)

                for j in range(len(files)):
                    key = datasets[i] + '_' + files[j][:-4]

                    lb = valid.ims[key + '_hr']
                    lb_tensor = torch.Tensor(np.expand_dims(
                        np.transpose(lb, [2, 0, 1]), axis=0)).cuda()
                    
                    input_im = valid.ims[key + '_' + scale]
                    m = torch.Tensor(valid.ims[key + '_' + scale + '_matrix']).double().cuda()
                    if "PreUpsample" in opt.valWDir:
                        post = torch.Tensor([
                            [0.5, 0, -0.25],
                            [0, 0.5, -0.25],
                            [0, 0, 1],
                        ])
                        m = torch.matmul(m, post.double().cuda())
                        
                    input_im = input_im.astype(np.float32) / 255.0
                    im = torch.Tensor(np.expand_dims(
                        np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()
                    
                    if opt.twoStage:
                        feat_im = mulut_predict(model_G, im, 1, 'valid', opt)
                        hyper_in = feat_im / float(opt.norm)
                    else:
                        feat_im = torch.round(im * opt.norm)
                        hyper_in = im

                    pred_hyper = mulut_predict(model_G, hyper_in, 2, 'valid', opt)

                    # get mask
                    all_white = torch.zeros_like(feat_im)
                    # shrave borders
                    border = 4
                    h, w = all_white.shape[-2:]
                    all_white[:,:,border:h-border,border:w-border] = 255

                    nn_warper.set_shape(feat_im.shape, m, lb_tensor.shape)
                    mask_output = nn_warper.warp(all_white).bool()
                    # mask_tensor = torch.Tensor(mask_output).bool()

                    # set shape per image
                    valid_warper.set_shape(feat_im.shape, m, lb_tensor.shape)
                    if opt.linear:
                        # linear
                        pred = valid_warper.warp(feat_im, pred_hyper)
                    else:
                        # Gaussian
                        if opt.inC == 1:
                            _, C, _, _ = pred_hyper.shape
                            hyper1_c_idx = list(range(0, C, 3))
                            hyper2_c_idx = list(range(1, C+1, 3))
                            hyper3_c_idx = list(range(2, C+2, 3))

                            pred = valid_warper.warp(feat_im, pred_hyper[:, hyper1_c_idx, :, :], pred_hyper[:, hyper2_c_idx, :, :], pred_hyper[:, hyper3_c_idx, :, :]).float()
                        else:
                            pred  = valid_warper.warp(feat_im, pred_hyper[:, :1*opt.featC, :, :], pred_hyper[:, 1*opt.featC:2*opt.featC, :, :], pred_hyper[:, 2*opt.featC:, :, :]).float()
                    pred[pred.isnan()] = 0
                    # print(feat_im.shape, pred_hyper.shape, pred.shape)

                    pred = torch.round(pred.clip(0, 255))
                    
                    # print(pred.shape, lb_tensor.shape, mask_output.shape)
                    psnr = mPSNR(pred, lb_tensor, mask_output, 255)
                    psnrs.append(psnr.item())

                    if iter < 10000 and "DIV2K" not in datasets[i]: # save input and gt at start
                        input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                        Image.fromarray(input_img).save(
                            os.path.join(result_path, '{}_input.png'.format(key.split('_')[-1])))
                        Image.fromarray(lb.astype(np.uint8)).save(
                            os.path.join(result_path, '{}_gt.png'.format(key.split('_')[-1])))
                        
                    pred_hyper = np.transpose(np.squeeze(pred_hyper.data.cpu().numpy(), 0), [1, 2, 0])
                    
                    if pred_hyper.shape[-1]==9 and opt.debug:
                        H, W, C = pred_hyper.shape
                        for c in range(C):
                            a = pred_hyper[:, :, c]
                            heat = np.zeros([H, W, 3])
                            heat[:, :, c//3] = np.clip(np.round(a * 255.0), 0, 255)
                            Image.fromarray(heat.astype(np.uint8)).save(os.path.join(result_path, "{}_{}_heat_{}.png".format(key.split('_')[-1], opt.name, str(c))))
                    if opt.featC == 3:
                        feat = np.transpose(np.squeeze(
                            feat_im.data.cpu().numpy(), 0), [1, 2, 0])
                        feat = np.round(np.clip(feat, 0, 255)).astype(np.uint8)
                        Image.fromarray(feat).save(
                            os.path.join(result_path, '{}_{}_feat.png'.format(key.split('_')[-1], opt.name)))

                    
                    if "DIV2K" not in datasets[i]:
                        
                        pred = np.clip(np.round(pred[0].data.cpu().numpy()), 0, 255).astype(np.uint8).transpose((1, 2, 0))
                        mask_output = (mask_output[0]*255).cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)

                        all_white_gt = (np.ones_like(np.array(lb))*255).astype(np.uint8)
                        pred = pred * np.array(mask_output==255) + np.array(mask_output!=255) * all_white_gt
                        np.save(os.path.join(result_path, "{}_{}_pred_hyper.npy".format(key.split('_')[-1], opt.name)), pred_hyper)
                        Image.fromarray(pred).save(
                            os.path.join(result_path, '{}_{}_output.png'.format(key.split('_')[-1], opt.name)))

                
                avg_psnr = np.mean(np.asarray(psnrs))
                metric_list.append("{:.2f}".format(avg_psnr))

                if not opt.lutft:
                    writer.add_scalar('mPSNR_{}/{}'.format(scale, datasets[i]), avg_psnr, iter)
                    writer.flush()

            logger.info('\t'.join(metric_list))



if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()
    
    # opt.trainDir = "../data/DIV2K"
    # opt.valDir = "../data/SRBenchmark"

    ### Tensorboard for monitoring ###
    if not opt.lutft:
        writer = SummaryWriter(log_dir=opt.expDir)
        logger_name = 'train'
    else:
        logger_name = 'lutft'

    logger_info(logger_name, os.path.join(opt.expDir, logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(opt_inst.print_options(opt))

    model = getattr(model, opt.model)

    model_G = model(opt, inC=opt.inC, outC=opt.outC).cuda()
    # print(model_G)

    if opt.linear:
        train_resizer = AmplifiedLinearResize2dTorch(support_sz=opt.suppSize, device=torch.device("cuda"))
        train_resizer.set_shape([opt.batchSize, 1, opt.cropSize, opt.cropSize], scale_factors=opt.scale)
        valid_resizer = AmplifiedLinearResize2dTorch(support_sz=opt.suppSize, device=torch.device("cuda"))
        valid_warper = AmplifiedLinearWarp2dTorch(support_sz=opt.suppSize, device=torch.device("cuda"))
    else:
        # Gaussian
        train_resizer = SteeringGaussianResize2dTorch(support_sz=opt.suppSize, device=torch.device("cuda"), max_sigma=opt.maxSigma)
        train_resizer.set_shape([opt.batchSize, 1, opt.cropSize, opt.cropSize], scale_factors=opt.scale)
        valid_resizer = SteeringGaussianResize2dTorch(support_sz=opt.suppSize, device=torch.device("cuda"), max_sigma=opt.maxSigma)
        valid_warper = SteeringGaussianWarp2dTorch(support_sz=opt.suppSize, device=torch.device("cuda"), max_sigma=opt.maxSigma)

    nn_warper = NearestWarp2dTorch(device=torch.device("cuda"))
    
    if opt.gpuNum > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=list(range(opt.gpuNum)))

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=opt.lr0, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weightDecay, amsgrad=False)

    # LR
    if opt.lr1 < 0:
        lf = lambda x: (((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = opt.lr1 / opt.lr0
        lr_a = 1 - lr_b
        lf = lambda x: (((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Load saved params
    if opt.startIter > 0:
        model_path = os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.startIter))
        lm = torch.load(model_path)
        model_G.load_state_dict(lm.state_dict(), strict=True)
        print("load Model from:", model_path)
        
        opt_path = os.path.join(opt.expDir, 'Opt_{:06d}.pth'.format(opt.startIter))
        
        if os.path.exists(opt_path):
            lm = torch.load(opt_path)
            opt_G.load_state_dict(lm.state_dict())
            print("load Optimizer from:", opt_path)

    train_iter = Provider(opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize, opt.nsigma, inC=opt.inC)
    
    rand_scale = np.arange(1, 4, 0.1)

    datasets, datasetsW = ['Set5'], ['Set5']

    valid = MultiSRBenchmark(opt.valDir, datasets, nsigma=opt.nsigma)
    validw = SRBenchmarkW(opt.valWDir, datasetsW)
    
    # Some preparations
    l_accum = [0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    # TRAINING
    i = opt.startIter
        
    for i in range(opt.startIter+1, opt.totalIter+1):
        model_G.train()

        # Data preparing
        st = time.time()
        im, lb = train_iter.next()
        im = im.cuda()
        lb = lb.cuda()
        
        dT += time.time() - st

        # TRAIN G
        st = time.time()
        opt_G.zero_grad()

        if opt.twoStage:
            feat_im = mulut_predict(model_G, im, 1, 'train', opt)
            hyper_in = feat_im/float(opt.norm)
            # feat_im: [0, 255]
            # hyper_in: [0, 1]
        else:
            feat_im = torch.round(im * opt.norm)
            hyper_in = im

        pred_hyper = mulut_predict(model_G, hyper_in, 2, 'train', opt)
        # print(pred_hyper[0, 0, :8, :8].data)


        if opt.linear:
            pred = train_resizer.resize(feat_im, pred_hyper)
        else:
            pred  = train_resizer.resize(feat_im, pred_hyper[:, :1*opt.featC, :, :], pred_hyper[:, 1*opt.featC:2*opt.featC, :, :], pred_hyper[:, 2*opt.featC:, :, :])
        
        # print(im.shape, feat_im.shape, pred_hyper.shape, pred.shape, lb.shape)
        pred = torch.clamp(pred, 0, opt.norm) / float(opt.norm)

        loss_G = F.mse_loss(pred, lb)

        loss_G.backward()
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += opt.batchSize
        l_accum[0] += loss_G.item()

        # Show information
        if i % opt.displayStep == 0:
            if not opt.lutft:
                writer.add_scalar('loss_Pixel', l_accum[0] / opt.displayStep, i)

            logger.info("{} | Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                opt.expDir, i, accum_samples, l_accum[0] / opt.displayStep, dT / opt.displayStep,
                                                           rT / opt.displayStep))
            l_accum = [0., 0., 0.]
            dT = 0.
            rT = 0.

        # Save models
        if i % opt.saveStep == 0 and (not opt.lutft):
            if opt.gpuNum > 1:
                SaveCheckpoint(model_G.module, opt_G, opt, i)
            else: 
                SaveCheckpoint(model_G, opt_G, opt, i)

        # Validation
        if (i % opt.valStep == 0) or (opt.debug and i == 1):
            # validation during multi GPU training

            if opt.gpuNum > 1:
                valid_steps_warp(model_G.module, validw, opt, i)
                valid_steps(model_G.module, valid, opt, i)
            else:
                valid_steps_warp(model_G, validw, opt, i)
                valid_steps(model_G, valid, opt, i)
            
    if opt.lutft:
        stage = 2
        # outC = opt.outC
        for mode in opt.modes2:
            for r in [0, 1]:
                key = "s{}_{}r{}".format(str(stage), mode, r)
                ft_lut_path = os.path.join(opt.expDir, "LUTft_{}.npy".format(key))
                lut_weight = np.round(np.clip(getattr(model_G, "weight_s{}_{}r{}".format(stage, mode, r)).cpu().detach().numpy(), -1, 1) * 127).astype(np.int8)
                np.save(ft_lut_path, lut_weight)

        stage = 1
        r = 0
        for mode in opt.modes:
            key = "s{}_{}r{}".format(str(stage), mode, r)
            ft_lut_path = os.path.join(opt.expDir, "LUTft_{}.npy".format(key))
            lut_weight = np.round(np.clip(getattr(model_G, "weight_s{}_{}r{}".format(stage, mode, 0)).cpu().detach().numpy(), -1, 1) * 127).astype(np.int8)
            np.save(ft_lut_path, lut_weight)

        logger.info("Finetuned LUT saved to {}".format(opt.expDir))
    logger.info("Complete")

