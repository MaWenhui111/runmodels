import argparse


def parse_args():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', '-e',
                        dest='EPOCHS',
                        type=int,
                        default=200,
                        help='Number of epochs')
    parser.add_argument('--batch-size', '-b',
                        dest='batch_size',
                        type=int,
                        default=24,
                        help='Batch size')
    parser.add_argument('--learning-rate', '-l',
                        dest='lr',
                        type=float,
                        default=1e-2,
                        help='Learning rate')
    parser.add_argument('--load', '-f',
                        type=str,
                        default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--scale', '-s',
                        type=float,
                        default=0.6,
                        help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v',
                        dest='val',
                        type=float,
                        default=0.2,
                        help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('--classes', '-c',
                        type=int,
                        default=2,
                        help='Number of classes')

    parser.add_argument('--model', '-m',
                        default='/sdata/wenhui.ma/program/models/SFNet-master/mobi_seg_checkpoints/best_model/checkpoint_epoch195.pth',
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i',
                        default='/sdata/wenhui.ma/program/dataset/Mobi-Seg/test/image/',
                        metavar='INPUT',
                        nargs='+',
                        help='Filenames of input images')
    parser.add_argument('--output', '-o',
                        default='/sdata/wenhui.ma/program/models/SFNet-master/mobi_seg_predict/',
                        metavar='OUTPUT',
                        nargs='+',
                        help='Filenames of output images')
    parser.add_argument('--save',
                        default=False,
                        help='Save the output masks')
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help='Use mixed precision')
    # parser.add_argument('--viz',
    #                     action='store_true',
    #                     help='Visualize the images as they are processed')
    parser.add_argument('--mask-threshold',
                        type=float,
                        default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear',
                        action='store_true',
                        default=False,
                        help='Use bilinear upsampling')

    return parser.parse_args()
