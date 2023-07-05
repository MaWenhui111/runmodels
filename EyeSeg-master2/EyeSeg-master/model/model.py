#!/usr/bin/env python3
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# import numpy as np

class EncoderBlock(nn.Module):
    """EncoderBlock Inherets the pytroch nn.Module class 

    EncoderBlock is a representation of a single instance of the encoder portion
    of the network. 

    Arguments:
        input_channels: the number of input channels that each convolutional layer
            will have, must be an integer. 
        output_channels: the number of output filters from each convolutional layer,
            must be an integer.
        down_size: used as a condition if the encoder block will be downsampling the 
            input, must be a boolean.
        prob: this will be a value to represent the percentage of dropout to apply at 
            each dropout layer instance, must be a floating point value.

    """
    def __init__(self,input_channels : int, output_channels : int, down_size : bool, prob : float, dilation=(1,1)):
        super(EncoderBlock, self).__init__()
        
        # Downsample boolean
        self.down_size = down_size

        # Convolution Layers
        self.conv1_1 = nn.Conv2d(input_channels
                                ,output_channels
                                ,kernel_size=(3,3)
                                ,padding=(1,1)
                                ,bias = False
                                )
        self.conv2_1 = nn.Conv2d(input_channels + output_channels
                                ,output_channels
                                ,kernel_size=(1,1)
                                ,padding=(0,0)
                                ,dilation=dilation
                                ,bias = False
                                )
        self.conv2_2 = nn.Conv2d(output_channels
                                ,output_channels
                                ,kernel_size=(3,3)
                                ,padding=(1,1)
                                ,bias = False
                                )
        self.conv3_1 = nn.Conv2d(input_channels + 2*output_channels
                                ,output_channels
                                ,kernel_size=(1,1)
                                ,padding=(0,0)
                                ,dilation=dilation
                                ,bias = False
                                )
        self.conv3_2 = nn.Conv2d(output_channels
                                ,output_channels
                                ,kernel_size=(3,3)
                                ,padding=(1,1)
                                ,bias = False
                                )
        
        # Pooling Layer
        self.max_pool = nn.MaxPool2d((2,2))
        
        # Activations
        self.prelu1 = nn.PReLU(num_parameters=output_channels)
        self.prelu2 = nn.PReLU(num_parameters=output_channels)
        self.prelu3 = nn.PReLU(num_parameters=output_channels)
        
        # Batch Norms
        self.bn1 = torch.nn.BatchNorm2d(num_features=output_channels, momentum=0.99)
        self.bn2 = torch.nn.BatchNorm2d(num_features=output_channels, momentum=0.99)
        

        # Dropout Layers
        self.dropout1 = nn.Dropout(prob)
        self.dropout2 = nn.Dropout(prob)
        self.dropout3 = nn.Dropout(prob)

    def forward(self, tensor_in : torch.Tensor) -> torch.Tensor:
        """ a forward or __call__ of the encoder module

        forward will process all the connecitons and calculations of a torch.Tensor object
        and provide feature maps/information through the network. This will be a repeated process
        from input to output to the decoder portion.

        Note: The main concepts of the block that we applied are the multiple stacks and
        residuals throughout each block and the network as a whole.

        Arguments:

            tensor: a Tensor object holding tf.float32 data of the 2D images in the form
                (batch_of_images, image_rgb_channels, image_rows, image_columns)
        """
        layer1_in = self.max_pool(tensor_in) if self.down_size else tensor_in
        layer1_1 = self.prelu1(self.bn1(self.conv1_1(layer1_in)))
        layer1_1 = self.dropout1(layer1_1)
        layer1_cat = torch.cat((layer1_in, layer1_1), dim=1)
        layer2_1 = self.dropout2(self.prelu2(self.conv2_2(self.conv2_1(layer1_cat))))
        layer2_cat = torch.cat((layer1_cat, layer2_1), dim=1)
        layer3_12 = self.prelu3(self.bn2(self.conv3_2(self.conv3_1(layer2_cat))))
        end_block = self.dropout3(layer3_12)
        return end_block
    
    
class DecoderBlock(nn.Module):
    """ The decoder block will be the upsampling portion of the encoder-decoder model

    This will be a repeated module within the encoder-decoder leveraging data from earlier portions 
    of the encoder modules, known as a skip conneciton or residual connection, and will be used in
    conjunction with the normal passthrough tensor along with periodic residual connections within
    this module itself.

    Notes: I may change to have an additional 2 convolutional layers to match a mirrored pattern
        of the encoder if it helps the inference of the class features.
    
    Arguments:

        skip_channels: this will be a tensor of a previous encoder module that isnt explicitly 
            before this module, must be a Tensor Object.
        
        input_channels: the number of filters each convolution will have, must be an integer.
        
        output_channels: the number of filters outbound of each convolution within this module
            and this must also be an integer.

        up_stride: this will be the value for how much we upsample the image at each module
            this must be in integer.
        
        prob: this will be a value to represent the percentage of dropout to perform, this will be a float.
    """
    def __init__(self, skip_channels : int, input_channels : int, output_channels : int, up_stride : int, prob : float):
        super(DecoderBlock, self).__init__()

        # Upsampling metric
        self.up_sample = up_stride

        # Convolution Layers
        self.conv1_1 = nn.Conv2d(skip_channels+input_channels
                                ,output_channels
                                ,kernel_size=(1,1)
                                ,padding=0
                                ,bias = False
                                )
        self.conv1_2 = nn.Conv2d(input_channels
                                ,output_channels
                                ,kernel_size=(3,3)
                                ,padding=1
                                ,bias = False
                                )
        self.conv2_1 = nn.Conv2d(3*input_channels
                                ,output_channels
                                ,kernel_size=(1,1)
                                ,padding=0
                                ,bias = False
                                )
        self.conv2_2 = nn.Conv2d(input_channels
                                ,output_channels
                                ,kernel_size=(3,3)
                                ,padding=1
                                ,bias = False
                                )

        # Dropout Layers
        self.dropout1 = nn.Dropout(prob)
        self.dropout2 = nn.Dropout(prob)
        
        # Activations
        self.leaky_relu1 = nn.LeakyReLU()
        self.leaky_relu2 = nn.LeakyReLU()

    def forward(self, prev_feature_map : torch.Tensor, tensor_in : torch.Tensor) -> torch.Tensor:
        """ a forward or __call__ of the decoder module

        Notes:

        Arguments:

        """
        up_sampled = nn.functional.interpolate(tensor_in,scale_factor=self.up_sample,mode='nearest')
        # print(up_sampled.shape, prev_feature_map.shape)
        skip_connection_cat = torch.cat((up_sampled, prev_feature_map), dim = 1)
        layer1 = self.leaky_relu1(self.conv1_2(self.conv1_1(skip_connection_cat)))
        layer1 = self.dropout1(layer1)
        layer1_cat = torch.cat((skip_connection_cat, layer1), dim = 1)
        layer2 = self.leaky_relu2(self.conv2_2(self.conv2_1(layer1_cat)))
        out = self.dropout2(layer2)
        return out


    
class EyeSeg(nn.Module):
    """ This will be the overall neural netowrk in an encoder-decoder format

    EyeSeg Arcitecture

    https://openeyes-workshop.github.io/downloads/openeyes2020_jonathan_perry_eyeseg_fast_and_efficient_few_shot_semantic_segmentation.pdf

    The structure should be the following:


    """
    def __init__(self, in_channels : int, out_channels : int, channel_size : int, prob : float):
        super(EyeSeg, self).__init__()

        self.down_block1 = EncoderBlock(input_channels=in_channels
                                        ,output_channels=channel_size
                                        ,down_size=False
                                        ,prob=prob
                                        )
        self.down_block2 = EncoderBlock(input_channels=channel_size
                                        ,output_channels=channel_size
                                        ,down_size=True
                                        ,prob=prob
                                        ,dilation=(2,2)
                                        )
        self.down_block3 = EncoderBlock(input_channels=channel_size
                                        ,output_channels=channel_size
                                        ,down_size=True
                                        ,prob=prob
                                        ,dilation=(4,4)
                                        )
        self.down_block4 = EncoderBlock(input_channels=channel_size
                                        ,output_channels=channel_size
                                        ,down_size=True
                                        ,prob=prob
                                        )

        self.down_block5 = EncoderBlock(input_channels=channel_size
                                        ,output_channels=channel_size
                                        ,down_size=True
                                        ,prob=prob
                                        )

        self.up_block1 = DecoderBlock(skip_channels=channel_size
                                     ,input_channels=channel_size
                                     ,output_channels=channel_size
                                     ,up_stride=(2,2)
                                     ,prob=prob
                                     )
        self.up_block2 = DecoderBlock(skip_channels=channel_size
                                     ,input_channels=channel_size
                                     ,output_channels=channel_size
                                     ,up_stride=(2,2)
                                     ,prob=prob
                                     )
        self.up_block3 = DecoderBlock(skip_channels=channel_size
                                     ,input_channels=channel_size
                                     ,output_channels=channel_size
                                     ,up_stride=(2,2)
                                     ,prob=prob
                                     )

        self.up_block4 = DecoderBlock(skip_channels=channel_size
                                     ,input_channels=channel_size
                                     ,output_channels=channel_size
                                     ,up_stride=(2,2)
                                     ,prob=prob
                                     )

        self.out_conv = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels, momentum=0.99)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, tensor_in : torch.Tensor) -> torch.Tensor:
        """ a forward or __call__ for EyeSeg
        """
        downblock1 = self.down_block1(tensor_in)
        downblock2 = self.down_block2(downblock1)
        downblock3 = self.down_block3(downblock2)
        downblock4 = self.down_block4(downblock3)
        downblock5 = self.down_block5(downblock4)
        upblock1 = self.up_block1(downblock4, downblock5)
        upblock2 = self.up_block2(downblock3, upblock1)
        upblock3 = self.up_block3(downblock2, upblock2)
        upblock4 = self.up_block4(downblock1, upblock3)
        out = self.bn(self.out_conv(upblock4))
        return out

if __name__ == '__main__':
    model = EyeSeg(1,4,32,0.30)
    summary(model,(1,640,400), device='cpu')  # (1,640,400)为输入的图片尺寸