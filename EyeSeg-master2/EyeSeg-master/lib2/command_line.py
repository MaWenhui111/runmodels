#!/usr/bin/env python3
from pprint import pprint
import argparse

class command_line_args:
    """
    """
    def parse_args(self):
        """
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-name'
                            , dest='MODEL_NAME'
                            , type=str
                            ,default= 'EyeSeg/'
                            )
        parser.add_argument('--learn-rate'
                            , dest='LEARN_RATE'
                            , type=float
                            ,default= 1e-3
                            )
        parser.add_argument('--cap-learn-rate'
                            , dest='CAP_LEARN_RATE'
                            , type=float
                            ,default= 1e-4
                            )
        parser.add_argument('--batch-size'   
                            , dest='BATCH_SIZE'
                            , type=int
                            , default = 32
                            )
        parser.add_argument('--epochs'       
                            , dest='EPOCHS'
                            , type=int
                            ,help='Number of epochs'
                            ,default= 200
                            )
        parser.add_argument('--save-path'    
                            , dest='SAVE_PATH'
                            , type=str
                            ,default='./model/saves/'
                            )
        parser.add_argument('--load-model'   
                            , dest='PRETRAINED_PATH'
                            , type=str
                            , default='./best_model.hdf5'
                            )
        parser.add_argument('--useGPU'       
                            , dest='USE_GPU'
                            , type=bool
                            , default=True
                            )
        parser.add_argument('--dist-training'
                            , dest='DIST_TRAIN'
                            , type=bool
                            , default=True
                            )
        parser.add_argument('--input-shape'  
                            , dest='INPUT_SHAPE'
                            , type=lambda s: tuple([int(token) for token in s.split(',')])
                            , default='96,128,1'
                            )
        parser.add_argument('--output-shape' 
                            , dest='OUTPUT_SHAPE'
                            , type=lambda s: tuple([int(token) for token in s.split(',')])
                            , default='512,1024,20'
                            )
        parser.add_argument('--num-classes'  
                            , dest='NUM_CLASSES'
                            , type=int
                            , default=2
                            )
        parser.add_argument('--decode-type'  
                            , dest='DECODE_TYPE'
                            , type=str
                            , default='utf-8'
                            )
        parser.add_argument('--train-images' 
                            , dest='TRAIN_IMAGES'
                            , type=str
                            , default='./data/train_images.txt'
                            )
        parser.add_argument('--ul-images' 
                            , dest='UL_IMAGES'
                            , type=str
                            , default='./data/ul_images.txt'
                            )
        parser.add_argument('--train-labels' 
                            , dest='TRAIN_LABELS'
                            , type=str
                            , default='./data/train_labels.txt'
                            )
        parser.add_argument('--val-images'   
                            , dest='VAL_IMAGES'
                            , type=str
                            , default='./data/val_images.txt'
                            )
        parser.add_argument('--val-labels'   
                            , dest='VAL_LABELS'
                            , type=str
                            , default='./data/val_labels.txt'
                            )
        parser.add_argument('--training'     
                            , dest='TRAIN'
                            , type=bool
                            , default=True
                            )
        parser.add_argument('--use-dropout'  
                            , dest='DROPOUT'
                            , type=bool
                            , default=True
                            )
        parser.add_argument('--dilation'  
                            , dest='DILATED'
                            , type=bool
                            , default=True
                            )
        parser.add_argument('--filter-size'  
                            , dest='FILTER_SIZE'
                            , type=int
                            , default=32
                            )
        parser.add_argument('--filters'  
                            , dest='FILTERS'
                            , type=int
                            , default=32
                            )
        parser.add_argument('--dropout-rate'  
                            , dest='DROPOUT_RATE'
                            , type=float
                            , default=0.20
                            )
        parser.add_argument('--root-folder'  
                            , dest='ROOT_FOLDER'
                            , type=str
                            , default='/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/'
                                      # './data/'
                            )
        parser.add_argument('--save-folder'  
                            , dest='SAVE_FOLDER'
                            , type=str
                            , default='./model/saves/'
                            )

        parser.add_argument('--label-color'  
                            , dest='LABEL_COLOR'
                            , type=str
                            , default='./data/label_colors.pkl'
                            )
        parser.add_argument('--workers'  
                            , dest='WORKERS'
                            , type=int
                            , default=4
                            )
        parser.add_argument('--framework'  
                            , dest='FRAMEWORK'
                            , type=str
                            , default='tf.keras'
                            )
        parser.add_argument('--retrain'  
                            , dest='RETRAIN'
                            , type=bool
                            , default=False
                            )
        parser.add_argument('--batch-iters'  
                            , dest='BATCH_ITERS'
                            , type=int
                            , default=2
                            )
        parser.add_argument('--log-file'  
                            , dest='LOG_FILE'
                            , type=str
                            , default='training.log'
                            )
        parser.add_argument('--epoch-start'  
                            , dest='EPOCH_START'
                            , type=int
                            , default=0
                            )
        parser.add_argument('--log-epoch'  
                            , dest='LOG_EPOCH'
                            , type=int
                            , default=1
                            )
        parser.add_argument('--mode'  
                            , dest='MODE'
                            , type=str
                            , default='semi'
                            )
        return parser.parse_args()


if __name__ == '__main__':
    arg = command_line_args().parse_args()
    arg.FRAMEWORK = ' Torch 2 '
    print(arg.FRAMEWORK.lower().strip())
    print(arg.INPUT_SHAPE)
