import copy

import cv2
import torch
import torchvision
import torch.nn.functional as F

import sys
#sys.path.append(r'D:\Users\userLittleWatermelon\codes\CE_Net_example\CE_Net_example')
sys.path.append('sdata/wenhui.ma/program/IrisUnet/src/')


from src.models.net_UNet import UNet
from src.models.base_model import BaseModel
from src.models.loss import *
from src.util.tools import *
import src.models.loss as ml
import src.util.tools as tl

#类似控制台，使用哪个网络模型

class IrisNet(BaseModel):
    def __init__(self, cfg, log_writer=None):
        super(IrisNet, self).__init__(cfg)

        self.cfg = cfg

        if cfg.method == 'UNet':
            self.model = UNet(n_classes=2, backbone='vgg16', pretrained= cfg.model_home_path[cfg.backbone])
            #cfg.backbone = 'vgg16'


        else:
            pass  # 出错

        if self.isTrain:
            # define loss function
            self.criterion = ml.Mask_Loss(seg_weight=1)

            self.optimizer_A = torch.optim.Adam(self.model.parameters(), lr=cfg.initial_lr, betas=cfg.betas,
                                                weight_decay=cfg.weight_decay)
           # self.optimizer_S = torch.optim.SGD(self.model.parameters(), lr=cfg.initial_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

            self.optimizers.append(self.optimizer_A)
           # self.optimizers.append(self.optimizer_S)

        self.model_names = ['model']
        self.loss_names = ['sum','mask']
        #self.loss_names = ['sum', 'mask', 'edge']

        if self.isTrain and cfg.lr_policy != 'poly':
            self.schedulers = [tl.get_scheduler(optimizer, cfg) for optimizer in self.optimizers]


    def update_loss_weight(self, seg_weight):
        self.criterion.seg_weight = seg_weight


    def set_input(self, input, device):
        self.raw_image = input['raw_image'].to(device)


        if self.isTrain:
            self.gt_mask_image = input['mask_image'].to(device)



    def forward(self):
        self.seg_map = self.model(self.raw_image)

        if not isinstance(self.seg_map, list):
            self.seg_map = [self.seg_map]


        seg = F.softmax(self.seg_map[-1], dim=1)
        self.seg = seg.argmax(dim=1).unsqueeze(1).float()  # N*1*h*w



    def process(self):
        self.forward()
        loss_list = self.criterion(self.seg_map, self.gt_mask_image)
        self.loss_sum, self.loss_mask= loss_list


    def optimize_parameters(self):
        self.process()
        self.optimizer_A.zero_grad()  # set H's gradients to zero
        self.loss_sum.backward()
        self.optimizer_A.step()  # update H's weights

    def show_eval_results(self, train_name, epoch, log_writer):

        org_img = tl.restore_img(tl.im_to_numpy(self.raw_image[0]))
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

        img_list = []
        input_img_list = [tl.im_to_torch(org_img), tl.gray2rgb(self.gt_mask_image[0]).cpu()]

        pre_img_list = [tl.im_to_torch(org_img), tl.gray2rgb(self.seg[0]).cpu()]

        img_list.extend(input_img_list)
        img_list.extend(pre_img_list)

        x = torchvision.utils.make_grid(img_list, nrow=3, padding=6, normalize=True, scale_each=True, pad_value=1)
        log_writer.add_image('{}-{}_input_VS_pre'.format(train_name, epoch), x)

