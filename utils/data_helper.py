#%%
import cv2
import numpy
from PIL import Image, ImageDraw

def read_text_data(file_name):
    """
    result example:
    [ [    '/media/zxf/Data/document_ML/ML/yolo/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg',
           ['263', '211', '324', '339', '8'],
           ['165', '264', '253', '372', '8'],
           ['5', '244', '67', '374', '8'],
           ['241', '194', '295', '299', '8'],
           ['277', '186', '312', '220', '8\n'] ],
      [    '/media/zxf/Data/document_ML/ML/yolo/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000007.jpg',
           ['141', '50', '500', '330', '6\n'] ] 
    ]
    """
    f = open(file_name,'r')
    lines = f.readlines()
    data_list = []
    for line in lines:
        line_sep           = line.split(' ')
        image_path         = line_sep[0]
        bunding_boxes_info = line_sep[1:]
        num_boxes          = int(len(bunding_boxes_info) / 5)
        bunding_boxes      = []
        bunding_boxes.append(image_path)
        for i in range(num_boxes):
            box = bunding_boxes_info[i*5:(i+1)*5]
            bunding_boxes.append(box)
        data_list.append(bunding_boxes)

    return data_list

def draw_boxes(img_path,bunding_boxes):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    for i in range(len(bunding_boxes)):
        box = bunding_boxes[i][0:4]
        draw.rectangle(( (int(box[0]), int(box[1])), (int(box[2]), int(box[3])) ))
    img.show()
