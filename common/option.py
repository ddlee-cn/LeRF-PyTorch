import argparse
import os
import pickle
import shutil
from pathlib import Path


class BaseOptions():
    def __init__(self, debug=False):
        self.initialized = False
        self.debug = debug

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='lerf',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='SRNetsSWF2')
        parser.add_argument('--scale', '-r', type=str, default='4')
        parser.add_argument('--nsigma', type=int, default=-1)
        parser.add_argument('--nf', type=int, default=64)
        parser.add_argument('--modes', type=str, default='sct')
        parser.add_argument('--modes2', type=str, default='sct')
        parser.add_argument('--interval', type=int, default=4, help='N bit uniform sampling')
        parser.add_argument('--norm', type=int, default=255)
        parser.add_argument('--suppSize', type=int, default=2)
        parser.add_argument('--inC', type=int, default=1, help='training sample channel')
        parser.add_argument('--outC', type=int, default=3, help='hyper stage output channel')
        parser.add_argument('--featC', type=int, default=1, help='Net input and feat set to three channels')
        parser.add_argument('--maxSigma', type=int, default=10)
        parser.add_argument('--stages', type=int, default=2, help="repeat block number in feature extraction stage")
        parser.add_argument('--twoStage', action='store_true', default=False, help="w/ pre-filter stage")
        parser.add_argument('--linear', action='store_true', default=False, help="linear resampling function")
        parser.add_argument('--modelRoot', type=str, default='./models')

        parser.add_argument('--expDir', '-e', type=str, default='')
        parser.add_argument('--load_from_opt_file', action='store_true', default=False)

        parser.add_argument('--debug', default=False, action='store_true')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        if self.debug:
            opt = parser.parse_args("")
        else:
            opt = parser.parse_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def save_options(self, opt):
        file_name = os.path.join(opt.valoutDir, 'opt')
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def process(self, opt):
        # opt.modelRoot = os.path.join(opt.modelRoot, opt.task)
        if "dn" in opt.task:
            opt.flag = opt.sigma
        elif "db" in opt.task:
            opt.flag = opt.qf
        elif "sr" in opt.task:
            opt.flag = opt.scale
        else:
            opt.flag = "0"
        return opt

    def save_code(self):
        src_dir = "./"
        trg_dir = os.path.join(self.opt.expDir, "code")
        for f in Path(src_dir).rglob("*.py"):
            trg_path = os.path.join(trg_dir, f)
            os.makedirs(os.path.dirname(trg_path), exist_ok=True)
            shutil.copy(os.path.join(src_dir, f), trg_path, follow_symlinks=False)

    def parse(self, save=False):
        opt = self.gather_options()

        opt.isTrain = self.isTrain  # train or test

        opt = self.process(opt)
        if "." not in opt.scale:
            opt.scale = int(opt.scale)
        else:
            # downsample
            opt.scale = float(opt.scale)

        if opt.expDir == '':
            opt.modelDir = os.path.join(opt.modelRoot, opt.name)

            if not os.path.isdir(opt.modelDir):
                os.mkdir(opt.modelDir)

            count = 1
            while True:
                if os.path.isdir(os.path.join(opt.modelDir, 'expr_{}'.format(count))):
                    count += 1
                else:
                    break
            opt.expDir = os.path.join(opt.modelDir, 'expr_{}'.format(count))
            os.mkdir(opt.expDir)
        else:
            if not os.path.isdir(opt.expDir):
                os.makedirs(opt.expDir)
            opt.name = opt.expDir.split("/")[-1] + "-" + opt.model

        opt.modelPath = os.path.join(opt.expDir, "Model.pth")

        if opt.isTrain:
            opt.valoutDir = os.path.join(opt.expDir, 'val')
            if opt.lutft:
                opt.valoutDir = os.path.join(opt.expDir, 'lutft')
            if not os.path.isdir(opt.valoutDir):
                os.mkdir(opt.valoutDir)
            self.save_options(opt)

        # self.print_options(opt)

        if opt.isTrain and opt.debug:
            opt.displayStep = 10
            opt.saveStep = 100
            opt.valStep = 50
            opt.totalIter = 200
            opt.batchSize = 4
            opt.nf = 16

        self.opt = opt

        if opt.isTrain and (not opt.debug):
            self.save_code()
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # data
        parser.add_argument('--batchSize', type=int, default=16)
        parser.add_argument('--cropSize', type=int, default=48, help='input LR training patch size')
        parser.add_argument('--cropSizeLR', type=int, default=48, help='input LR training patch size')
        parser.add_argument('--trainDir', type=str, default="./data/DIV2K")
        parser.add_argument('--valDir', type=str, default='./data/rrBenchmark')
        parser.add_argument('--valWDir', type=str, default='./data/WarpBenchmark')
        parser.add_argument('--lutft', action='store_true', default=False)
        # training
        parser.add_argument('--startIter', type=int, default=0,
                            help='Set 0 for from scratch, else will load saved params and trains further')
        parser.add_argument('--totalIter', type=int, default=50000, help='Total number of training iterations')
        parser.add_argument('--displayStep', type=int, default=100, help='display info every N iteration')
        parser.add_argument('--valStep', type=int, default=2000, help='validate every N iteration')
        parser.add_argument('--saveStep', type=int, default=2000, help='save models every N iteration')
        parser.add_argument('--lr0', type=float, default=1e-3)
        parser.add_argument('--lr1', type=float, default=1e-4)
        parser.add_argument('--weightDecay', type=float, default=0)
        parser.add_argument('--gpuNum', '-g', type=int, default=1)
        parser.add_argument('--workerNum', '-n', type=int, default=8)

        self.isTrain = True
        return parser

    def process(self, opt):
        return opt


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--testDir', type=str, default='./data/rrBenchmark')
        parser.add_argument('--resultRoot', type=str, default='./results')
        parser.add_argument('--loadIter', type=int, default=50000, help='Total number of training iterations')
        parser.add_argument("--lutName", type=str, default="LUTft")


        self.isTrain = False
        return parser

    def process(self, opt):
        # opt.lutPath = os.path.join(opt.expDir,
                                #    "Model_x{}_{}bit_int8.npy".format(opt.flag, opt.interval))
        # opt.resultDir = os.path.join(opt.resultRoot, opt.expDir.split("/")[-1], opt.testDir.split("/")[-1],
                                    #  "x{}_{}bit".format(opt.flag, opt.interval))
        # if not os.path.isdir(opt.resultDir):
            # os.makedirs(opt.resultDir)
        pass
        return opt
