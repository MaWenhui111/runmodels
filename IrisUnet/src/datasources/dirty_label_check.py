import glob
import os
import cv2


def check_signal(mask_img, process, seg_path=None, num_classes=4):
    img = mask_img
    if not ((img >= 0).all and(img < num_classes).all()):
        if seg_path is not None:
            print(os.path.basename(seg_path), process)
        else:
            print('与下data_transform同', process)


def check(root_path='/sdata/wenhui.ma/program/IrisUnet/data/Iris-Seg/', num_classes=2):
    data_path = os.path.join(root_path, 'train')

    path_list = [name for name in glob.glob(os.path.join(data_path, 'label', '*'))]
    name_list = [os.path.basename(name)
                 for name in glob.glob(os.path.join(data_path, 'label', '*'))]

    for i in range(len(name_list)):
        img = cv2.imread(path_list[i], cv2.IMREAD_GRAYSCALE)
        if not ((img >= 0).all and(img < num_classes).all()):
            print(name_list[i])


if __name__ == '__main__':
    check()