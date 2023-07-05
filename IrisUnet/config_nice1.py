import warnings


class DefaultConfig(object):
    def __init__(self):
        # Model structure
        self.name = 'IrisNet'
        self.model_arch = 'IrisNet'
        self.method = 'UNet'

        self.backbone = 'vgg16'
        # 预训练模型
        self.model_home_path = {
            'resnet18': '/data1/caiyong.wang/models/resnet_pytorch/resnet18-5c106cde.pth',
            'vgg16': '/sdata/wenhui.ma/program/IrisUnet/src/pth/vgg16_bn-6c64b313.pth'
        }

        # Training strategy
        self.isTrain = True

        self.num_workers = 4  # number of data loading workers
        #self.max_epochs = 2000  # number of total epochs to run
        self.max_epochs = 20  # number of total epochs to run

        self.train_batch = 10  # train batchsize
        self.val_batch = 6  # validation batchsize
        self.initial_lr = 2.0e-4 # initial learning rate
        self.lr_decay = 0.4  # when val_loss increase, lr = lr*lr_decay
        self.lr_policy = 'plateau' # poly
        self.weight_decay = 5e-4  # loss function
        self.momentum = 0.9
        self.power = 0.9  # poly
        self.log_interval = 10  # 这是啥
        self.betas = (0.9, 0.99)
        #self.device = "cuda:3"  # 设置gpu的序号
        self.device = "cuda:0"  # 设置gpu的序号

        # Data processing
        #self.dataset = "NICE"
        self.dataset = "Iris-Seg"  # "NDCLD15"
        # self.img_format = ['.tiff', '.png']
        self.img_format = ['.jpg', '.png']

        #self.root_path = "/data1/caiyong.wang/data/IrisLocSeg/" + self.dataset  # train dataset path
        #self.root_path = r'E:\files\use2Exp\NDCLD15\forSeg\spoof'
        # self.root_path = '/sdata/wenhui.ma/program/IrisUnet/data/forseg/'
        self.root_path = '/sdata/wenhui.ma/program/IrisUnet/data/Iris-Seg/'
        self.input_res = (448, 448)  # input size (h,w)
        self.seed = 17  # random seed

        # checkpoints
        self.checkpoint = "./checkpoints/"  # path to save checkpoints
        self.visible = True  # isTensorboard

    def parse(self, kwargs):
        """
        update cfg object
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        # print('user config:')
        # for k, v in self.__class__.__dict__.items():
        #     if not k.startswith('_'):
        #         print(k, getattr(self, k))

    def __str__(self):
        config = ""
        for name, value in vars(self).items():
            config += ('%s=%s\t\n' % (name, value))
        return config
