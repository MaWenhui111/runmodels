import configparser  # https://www.cnblogs.com/dion-90/p/7978081.html rw ini

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import lr_scheduler


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def restore_img(img):  # h*w*c
    # org = (img+1)*255/2   # from gaze
    img_copy = img.copy()  # prevent from change src
    mean = (0.406, 0.456, 0.485)  # BGR
    std = (0.225, 0.224, 0.229)
    org = img_copy * std
    org = org + mean
    org = org * 255.0
    return org.astype(np.uint8)


def restore_point_torch(points, h, w):  # points: N*2
    x = points[:, 0] * w
    y = points[:, 1] * h
    return torch.stack((x, y), dim=1)


def list_to_torch(input_list):
    tensor = to_torch(np.array(input_list)).float()
    return tensor


def gray2rgb(tensor):  # 1*h*w-->3*h*w
    assert tensor.shape[0] == 1
    return torch.cat([tensor, tensor, tensor], dim=0)


def crop_image(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (pad_height, pad_width)

    @return padded image
    """
    (pad_height, pad_width) = (int(pads[0]), int(pads[1]))
    height, width = img.shape[0:2]

    if len(img.shape) == 2:
        return img[0:height - pad_height, 0:width - pad_width]
    elif len(img.shape) == 3:
        return img[0:height - pad_height, 0:width - pad_width, :]
    else:
        return img


def resize_image(img, h, w):
    """
    img: numpy array of the shape (height, width)
    org_size: (height, width)

    @return resized image
    """
    img1 = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img1


def get_point(filename):
    with open(filename, 'r') as f:
        raw_data = f.readlines()
        points = []
        for data in raw_data[3:-1]:
            point = []
            for x in data.split()[:-1]:
                point.append(float(x))
            point.append(int(data.split()[-1]))
            points.append(point)
    return np.asarray(points)


def save_points(points_p, filename):  # points_p： 8*2
    file_header = 'version: 1\nn_points:  8\n{\n'
    f1 = open(filename, 'w')
    f1.write(file_header)
    for pt in points_p:
        msg = str(pt[0]) + ' ' + str(pt[1]) + ' ' + '1\n'
        f1.write(msg)
    f1.write('}')
    f1.close()


# input points : x,y ，N*2， image：BGR
def draw_points_on_image(points, image):
    img_copy = image.copy()
    if len(img_copy.shape) == 2:
        dst = cv2.merge([img_copy, img_copy, img_copy])
    else:
        dst = img_copy
    for point in points:
        cv2.circle(dst, (int(point[0]), int(point[1])), 3, (0, 255, 0), -2)  # -2 means solid circle
    return dst


def gen_new_ellipse_from_points(poins):  # 4*2
    p1 = poins[0, :]
    p2 = poins[1, :]
    p3 = poins[2, :]
    p4 = poins[3, :]
    cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
    cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0
    rl = np.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2) / 2.0
    rs = np.sqrt((p2[0] - p4[0]) ** 2 + (p2[1] - p4[1]) ** 2) / 2.0
    theta = np.arccos((p1[0] - p3[0]) / np.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2))
    return (cx, cy), rl, rs, theta


def show_train_middle_result(img, mask, pupil, iris):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    #    img = img.astype(np.uint8)[...,::-1] #https://blog.csdn.net/yuanlulu/article/details/79982347
    # image = image[:, :, ::-1] # BGR-->RGB
    plt.ion()
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.title("original image")
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.title("mask")
    plt.imshow(mask)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title("pupil")
    plt.imshow(pupil)
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title("iris")
    plt.imshow(iris)
    plt.axis('off')
    plt.tight_layout()
    plt.pause(5)  # print second
    plt.close('all')


def parse_ellipse_param(filename):
    target_param = configparser.ConfigParser()

    target_param.read(filename)

    iris_coordinate = (
        target_param['iris']['center_x'],
        target_param['iris']['center_y'],
        target_param['iris']['long_radius'],
        target_param['iris']['short_radius'],
        target_param['iris']['rad_phi']
    ) if target_param['iris']['exist'] == 'true' else (0, 0, 0, 0, 0)
    pupil_coordinate = (
        target_param['pupil']['center_x'],
        target_param['pupil']['center_y'],
        target_param['pupil']['long_radius'],
        target_param['pupil']['short_radius'],
        target_param['pupil']['rad_phi']
    ) if target_param['pupil']['exist'] == 'true' else (0, 0, 0, 0, 0)

    iris_coordinate = np.array(iris_coordinate, dtype=np.float32)
    pupil_coordinate = np.array(pupil_coordinate, dtype=np.float32)

    iris_exist = True if target_param['iris']['exist'] == 'true' else False
    pupil_exist = True if target_param['pupil']['exist'] == 'true' else False

    return iris_exist, iris_param, pupil_exist, pupil_param


def write_ellipse_param(filename, iris_exist, iris_param, pupil_exist, pupil_param):
    target_param = configparser.ConfigParser()
    target_param['iris'] = {}
    iris = target_param['iris']
    if iris_exist:
        iris['exist'] = 'true'
        iris['center_x'] = '%.2f' % iris_param.center[0]
        iris['center_y'] = '%.2f' % iris_param.center[1]
        iris['short_radius'] = '%.2f' % iris_param.second_radius
        iris['long_radius'] = '%.2f' % iris_param.first_radius
        iris['rad_phi'] = '%.2f' % iris_param.radian_phi
    else:
        iris['exist'] = 'false'

    target_param['pupil'] = {}
    pupil = target_param['pupil']
    if pupil_exist:
        pupil['exist'] = 'true'
        pupil['center_x'] = '%.2f' % pupil_param.center[0]
        pupil['center_y'] = '%.2f' % pupil_param.center[1]
        pupil['short_radius'] = '%.2f' % pupil_param.second_radius
        pupil['long_radius'] = '%.2f' % pupil_param.first_radius
        pupil['rad_phi'] = '%.2f' % pupil_param.radian_phi
    else:
        pupil['exist'] = 'false'

    with open(filename, 'w') as configfile:
        target_param.write(configfile)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, step, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if schedule and (step % schedule == 0):
        lr *= gamma
        # lr = max(lr, 1e-7)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9, ):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    new_lr = init_lr * (1 - iter / max_iter) ** power

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_decay, threshold=0.01,
                                                   patience=20)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


if __name__ == '__main__':
    iris_exist, iris_param, pupil_exist, pupil_param = parse_ellipse_param(
        'path')
    write_ellipse_param('test.ini', iris_exist, iris_param, pupil_exist, pupil_param)
