# %%
import cv2
import numpy as np
from PIL import Image, ImageDraw


#%%

path = "/media/zxf/Data/document_ML/ML/yolo/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000009.jpg"
image = cv2.imread(path)

#%%
#69 172 270 330 12
#150 141 229 284 14
#285 201 327 331 14
#258 198 297 329 14

img = Image.fromarray(image, 'RGB')
draw = ImageDraw.Draw(img)
draw.rectangle(((69, 172), (270, 330)))
draw.rectangle(((150, 141), (229, 284)))
draw.rectangle(((285, 201), (327, 331)))
draw.rectangle(((258, 198), (297, 329)))
img.show()
