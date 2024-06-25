import os
import threading
import time
import sys
import random

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "./")  # run under the project directory
from common.utils import modcrop

class Provider(object):
	def __init__(self, batch_size, num_workers, scale, path, patch_size, nsigma=-1, inC=1):
		self.data = DIV2K(scale, path, patch_size, nsigma, inC=inC)
		self.batch_size = batch_size
		self.num_workers = num_workers

		self.is_cuda = True
		self.data_iter = None
		self.iteration = 0
		self.epoch = 1

	def __len__(self):
		return int(sys.maxsize)

	def build(self):
		self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
		                                 shuffle=False, drop_last=False, pin_memory=False))

	def next(self):
		if self.data_iter is None:
			self.build()
		try:
			batch = self.data_iter.next()
			self.iteration += 1
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
			return batch[0], batch[1]
		except StopIteration:
			self.epoch += 1
			self.build()
			self.iteration += 1
			batch = self.data_iter.next()
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
			return batch[0], batch[1]


class DIV2K(Dataset):
    def __init__(self, scale, path, patch_size, nsigma=-1, inC=1, rigid_aug=True):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.inC = inC
        self.nsigma = nsigma
        self.file_list = [str(i).zfill(4)
                          for i in range(1, 801)]  # only using train images
        
        # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
        self.hr_cache = os.path.join(path, "cache_hr.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)
        
            
        if self.scale > 1:
            self.lr_cache = os.path.join(path, "cache_lr_x{}.npy".format(self.scale))
            if not os.path.exists(self.lr_cache):
                self.cache_lr()
                print("LR image cache to:", self.lr_cache)
            self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
            print("LR image cache from:", self.lr_cache)
        else:
            self.dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
            

    def cache_lr2(self):
        lr_dict = dict()
        dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
        for f in self.file_list:
            lr_dict[f] = np.array(Image.open(os.path.join(dataLR, f+"x{}.png".format(int(1/self.scale)))))
        np.save(self.lr_cache, lr_dict, allow_pickle=True)
        
    def cache_lr(self):
        lr_dict = dict()
        dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
        for f in self.file_list:
            lr_dict[f] = np.array(Image.open(os.path.join(dataLR, f+"x{}.png".format(self.scale))))
        np.save(self.lr_cache, lr_dict, allow_pickle=True)
        
    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "HR")
        for f in self.file_list:
            hr_dict[f] = np.array(Image.open(os.path.join(dataHR, f+".png")))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self.hr_ims[key]
        if self.scale > 1:
            im = self.lr_ims[key]
        else:
            # load LR on the fly
            im = np.array(Image.open(os.path.join(self.dataLR, key+"x{}.png".format(int(1/self.scale)))))

        shape = im.shape
        i = random.randint(0, shape[0]-self.sz)
        j = random.randint(0, shape[1]-self.sz)
        # lb = lb[i*self.scale:i*self.scale+self.sz*self.scale,
        #         j*self.scale:j*self.scale+self.sz*self.scale, :]
        lb = lb[int(i*self.scale):int(i*self.scale)+int(self.sz*self.scale),
                int(j*self.scale):int(j*self.scale)+int(self.sz*self.scale), :]
        im = im[i:i+self.sz, j:j+self.sz, :]

        if self.inC == 1:
            c = random.choice([0, 1, 2])
            im = im[:, :, c]
            lb = lb[:, :, c]


        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = lb.astype(np.float32)/255.0
        im = im.astype(np.float32)/255.0

        if self.inC == 1:
            lb = np.expand_dims(lb, axis=0)
            im = np.expand_dims(im, axis=0)
        else:
            lb = np.transpose(lb, [2, 0, 1])
            im = np.transpose(im, [2, 0, 1])

        if self.nsigma == 0:
             # blind noise
             noise_level = np.random.uniform(self.nsigma, self.max_nsigma)
             noise = np.random.normal(0, noise_level/255.0, im.shape).astype(np.float32)
        elif self.nsigma > 0:
             noise = np.random.normal(0, self.nsigma/255.0, im.shape).astype(np.float32)
        else:
            noise = 0
        
        im = im + noise

        return im, lb
    
    def __len__(self):
        return int(sys.maxsize)


class SRBenchmarkW(Dataset):
    def __init__(self, path, datasets):
        super(SRBenchmarkW, self).__init__()
        self.ims = dict()
        self.files = dict()
        self.datasets = datasets
        scales = ['isc', 'osc']
        # _ims_all = (5 + 14 + 100 + 100 + 109) * 2

        for dataset in datasets:
            folder = os.path.join(path, dataset, 'HR')
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(
                    os.path.join(path, dataset, 'HR', files[i])))
                # im_hr = modcrop(im_hr, scale)
                if len(im_hr.shape) == 2:
                    im_hr = np.expand_dims(im_hr, axis=2)

                    im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                key = dataset + '_' + files[i][:-4]
                self.ims[key + '_hr'] = im_hr

                for scale in scales:
                    im_lr = np.array(Image.open(
                        os.path.join(path, dataset, scale, files[i])))  # [:-4] + 'x%d.png'%scale)))
                    if len(im_lr.shape) == 2:
                        im_lr = np.expand_dims(im_lr, axis=2)
                        im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)
                    self.ims[key + '_' + scale] = im_lr

                    m = np.array(torch.load(os.path.join(path, dataset, scale, files[i].replace("png", "pth"))))

                    self.ims[key + '_' + scale + "_matrix"] = m


class MultiCustomSRBenchmark(Dataset):
    def __init__(self, path, datasets=['Set5', 'Set14', 'B100', 'Urban100', 'Manga109'], scale_pairs=[[2, 2], [3, 3], [4, 4]]):
        super(MultiCustomSRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        # _ims_all = (5 + 14 + 100 + 100 + 109) * 2

        for dataset in datasets:
            folder = os.path.join(path, dataset, 'HR')
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                for scale_p in scale_pairs:
                    scale_h, scale_w = scale_p
                    im_hr = np.array(Image.open(
                        os.path.join(path, dataset, 'HR', files[i])))
                    # im_hr = modcrop(im_hr, scale)
                    if len(im_hr.shape) == 2:
                        im_hr = np.expand_dims(im_hr, axis=2)

                        im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                    key = dataset + '_' + files[i][:-4] + 'hr'
                    self.ims[key] = im_hr

                    im_lr = np.array(Image.open(
                        os.path.join(path, dataset, 'LR_bicubic', "rrLR_X{:.2f}_{:.2f}".format(scale_h, scale_w), files[i])))  # [:-4] + 'x%d.png'%scale)))
                    if len(im_lr.shape) == 2:
                        im_lr = np.expand_dims(im_lr, axis=2)
                        im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)

                    key = dataset + '_' + files[i][:-4] + "X{:.2f}_{:.2f}".format(scale_h, scale_w)
                    self.ims[key] = im_lr

class MultiSRBenchmark(Dataset):
    def __init__(self, path, datasets=['Set5', 'Set14', 'B100', 'Urban100', 'Manga109'], scale_pairs=[[2, 2], [3, 3], [4, 4]], nsigma=-1):
        super(MultiSRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        self.datasets = datasets
        # _ims_all = (5 + 14 + 100 + 100 + 109) * 2

        for dataset in datasets:
            folder = os.path.join(path, dataset, 'HR')
            files = os.listdir(folder)
            files = [f for f in files if "png" in f]
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                for scale_p in scale_pairs:
                    scale_h, scale_w = scale_p
                    im_hr = np.array(Image.open(
                        os.path.join(path, dataset, 'HR', files[i])))
                    # im_hr = modcrop(im_hr, scale)
                    if len(im_hr.shape) == 2:
                        im_hr = np.expand_dims(im_hr, axis=2)

                        im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                    key = dataset + '_' + files[i][:-4] + 'hr'
                    self.ims[key] = im_hr

                    im_lr = np.array(Image.open(
                        os.path.join(path, dataset, 'LR_bicubic', "rrLR_X{:.2f}_{:.2f}".format(scale_h, scale_w), files[i])))
                    if len(im_lr.shape) == 2:
                        im_lr = np.expand_dims(im_lr, axis=2)
                        im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)

                    key = dataset + '_' + files[i][:-4] + "X{}".format(int(scale_h))
                    self.ims[key] = im_lr

