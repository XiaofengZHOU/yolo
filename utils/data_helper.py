#%%
import cv2
import numpy
from PIL import Image, ImageDraw

def read_text_data(file_name):
    """
    result example:
    [   ['/media/zxf/Data/document_ML/ML/yolo/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg',
        [263, 211, 324, 339, 8],
        [165, 264, 253, 372, 8],
        [5, 244, 67, 374, 8],
        [241, 194, 295, 299, 8],
        [277, 186, 312, 220, 8]],
        ['/media/zxf/Data/document_ML/ML/yolo/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000007.jpg',
        [141, 50, 500, 330, 6]]
    ]
    """
    f = open(file_name,'r')
    lines = f.readlines()
    data_list = []
    for line in lines:
        line_sep           = line.split(' ')
        image_path         = line_sep[0]
        bunding_boxes_info = [int(i) for i in line_sep[1:]]
        num_boxes          = int(len(bunding_boxes_info) / 5)
        bunding_boxes      = []
        bunding_boxes.append(image_path)
        for i in range(num_boxes):
            box = bunding_boxes_info[i*5:(i+1)*5]
            bunding_boxes.append(box)
        data_list.append(bunding_boxes)

    return data_list

def draw_boxes(bunding_boxes,img=None,img_path=None):
    """
    run example : draw_boxes(bunding_boxes_new,img=res)
    """
    if img_path!=None:
        img = Image.open(img_path)
    else:
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for i in range(len(bunding_boxes)):
        box = bunding_boxes[i][0:4]
        draw.rectangle(( (int(box[0]), int(box[1])), (int(box[2]), int(box[3])) ))
    img.show()


def reshape_image(bunding_boxes,img_size,img=None,img_path=None):
    """
    run example : res,bunding_boxes_new = reshape_image(data_list[0][1:],2000,img_path=data_list[0][0])
    """
    if img_path!=None:
        img = cv2.imread(img_path)
    height, width = img.shape[:2]
    res = cv2.resize(img,(img_size, img_size), interpolation = cv2.INTER_CUBIC)
    height_res, width_res = res.shape[:2]
    rate_height = height_res/height
    rate_width  = width_res/width

    bunding_boxes_new = []
    for box in bunding_boxes :
        box_new = [box[0]*rate_width,box[1]*rate_height,box[2]*rate_width,box[3]*rate_height,box[4]]
        bunding_boxes_new.append(box_new)

    return res,bunding_boxes_new


def get_labels(bunding_boxes):
    labels = []
    for box in bunding_boxes :
        x_center = (box[0]+ box[2])/2
        y_center = (box[1]+ box[3])/2
        box_w    = (box[2]- box[0])
        box_h    = (box[3]- box[1])
        class_id = box[4]
        label    = [x_center,y_center,box_w,box_h,class_id]
        labels.append(label)
    return labels
