## EyeSeg: Fast and Efficient Few-Shot Semantic Segmentation

#### Key Topics:

> OpenEDS 2020, Semantic Segmentation, lightweight model, real-time, encoder-decoder, Sparsely Annotated Data.


#### BibTex:
<pre>
@InProceedings{perryECCV2020EyeSeg,
author = {Perry, Jonathan and Fernandez, Amanda},
title = {EyeSeg: Fast and Efficient Few-Shot Semantic Segmentation},
booktitle = {European Conference on Computer Vision (ECCV) Workshops},
month = {Aug},
year = {2020}
}
</pre>

---

#### Model Architecture:
<img src="Utils/Network_Images/newest_model_skip.jpg"
     alt="Model Architecture"
     style="float: center; margin-right: 30px;" />

#### Train with OpenEDS 2020 Dataset for Sparse Semantic Segmentation:
<pre>python3 train.py --command-one=cmdone --command-two=cmdtwo</pre>

#### Requirements:
> Basic list of packages 
<pre>
matplotlib==3.2.1
numpy==1.18.3
opencv-python==4.2.0.34
Pillow==7.1.1
pprint==0.1
scikit-image==0.16.2
scikit-learn==0.22.2.post1
scipy==1.4.1
torch==1.10.1
torch-summary==1.3.2
torchsummary==1.5.1
torchvision==0.6.0
tqdm==4.46.1
</pre>
