import numpy as np
from PIL import Image


img = Image.open('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/train/image/999.jpg')
assert img.mode == 'RGB', "This image isn't in RGB mode"
r, g, b = img.split()
r = r.convert('RGB')
print(r)
img = np.asarray(img)
r = np.asarray(r)
print(r)
print(img.shape)