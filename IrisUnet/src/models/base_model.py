import os
import torch
import torchvision.utils
from collections import OrderedDict
from abc import ABC, abstractmethod
import sys
sys.path.append('/root/autodl-tmp/Iris_segment/CE_Net_example/CE_Net_example')
from src.util.osutils import mkdir_p, isfile, isdir, join
from src.util.tools import *
# from src.util.heatmap import *


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.isTrain = opt.isTrain
        self.save_dir = join(opt.checkpoint, opt.name, 'models')  # save all the checkpoints to save_dir
        if not isdir(self.save_dir):
            mkdir_p(self.save_dir)
        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.raw_image = None

    # self.metric = None # used for learning rate policy 'plateau'

    @abstractmethod
    def set_input(self, input, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
            device: device or devices
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def eval(self):
        """Make models eval mode during test/val time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()
    def freeze_bn(self):
        """Make model's BN freezed"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                for m in net.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()

    def to(self, device):
        """迁移模型到device"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.to(device)

    # def update_learning_rate(self):
    #     """Update learning rates for all the networks; called at the end of every epoch"""
    #     for scheduler in self.schedulers:
    #         scheduler.step(self.metric)

    def update_learning_rate(self, step, lr, schedule, gamma):
        """Update learning rates for all the networks; called at the end of every epoch"""
        lr_list = []
        for optimizer in self.optimizers:
            updated_lr = adjust_learning_rate(optimizer, step, lr, schedule, gamma)
            lr_list.append(updated_lr)

        return lr_list[-1] if len(lr_list) != 0 else lr

    def update_learning_rate_poly(self, step, last_lr, init_lr, max_iter, power):
        """Update learning rates for all the networks; called at the end of every epoch"""
        lr_list = []
        for optimizer in self.optimizers:
            updated_lr = poly_lr_scheduler(optimizer, init_lr, step, max_iter=max_iter, power=power)
            lr_list.append(updated_lr)

        return lr_list[-1] if len(lr_list) != 0 else last_lr

    def update_learning_rate_with_policy(self, metric):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step(metric)
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch, opt, device):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_%s_%s.pth' % (epoch, name)
        """
        state = OrderedDict([
            ('cfg', opt),
        ])
        save_filename = '%s_%s_%s.pth' % (epoch, opt.method, opt.dataset)
        save_path = os.path.join(self.save_dir, save_filename)

        # to cpu
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                state[name] = net.cpu().state_dict()  # use single gpu
        torch.save(state, save_path)

        # to gpu
        self.to(device)

    def load_networks(self, checkpoint):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if name in checkpoint.keys():
                    net.load_state_dict(checkpoint[name])

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
