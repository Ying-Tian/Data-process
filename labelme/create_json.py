import os
import base64
import numpy as np
import json
import argparse
import time
import cv2
from tqdm import tqdm
from skimage import measure, io


class CreateJson(object):
    def __init__(self):
        self.version = "3.16.7"
        self.flags = {}
        self.shapes = []
        self.lineColor = [0, 255, 0, 128]
        self.fillColor = [255, 0, 0, 128]
        self.imagePath = None
        self.imageData = None
        self.imageHeight = None
        self.imageWidth = None

    def to_json(self, name_list, input_folder):
        image_folder = os.path.join(input_folder, 'image')
        label_folder = os.path.join(input_folder, 'label')
        save_folder = os.path.join(input_folder, 'json')
        for name in tqdm(name_list):
            label_path = os.path.join(label_folder, name)
            image_path = os.path.join(image_folder, name)

            label = io.imread(label_path, as_gray=True)
            height, width = label.shape
            label[label > 0] = 255
            contours = measure.find_contours(label, 0.5)
            shapes = self.init_shape(contours)
            self.imageHeight = height
            self.imageWidth = width
            self.shapes = shapes

            self.imagePath = image_path
            self.image_to_byte(image_path)

            save_name = name.split('.')[0] + '.json'
            save_path = os.path.join(save_folder, save_name)
            self.save_json(save_path)

    def save_json(self, save_path):
        info_dict = dict()
        info_dict["version"] = self.version
        info_dict["flags"] = self.flags
        info_dict["shapes"] = self.shapes
        info_dict["lineColor"] = self.lineColor
        info_dict["fillColor"] = self.fillColor
        info_dict["imagePath"] = self.imagePath
        info_dict["imageData"] = self.imageData
        info_dict["imageHeight"] = self.imageHeight
        info_dict["imageWidth"] = self.imageWidth

        f = open(save_path, 'a')
        json.dump(info_dict, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        f.close()
        # print('save json to {}'.format(save_path))

    def image_to_byte(self, image_path):
        with open(image_path, 'rb') as f:
            img_byte = base64.b64encode(f.read()).decode('utf-8')
        self.imageData = img_byte

    def init_shape(self, contours):
        shape_dict = dict()
        shape_dict["label"] = "defect"
        shape_dict["line_color"] = None
        shape_dict["fill_color"] = None
        shape_dict["flags"] = self.flags
        shape_list = list()

        for n, contour in enumerate(contours):
            shape = shape_dict.copy()
            contour[:, [0, 1]] = contour[:, [1, 0]]
            approx_hand = measure.approximate_polygon(contour, tolerance=0.8)
            approx_hand = np.delete(approx_hand, -1, axis=0)
            if len(approx_hand) > 2:
                shape["shape_type"] = "polygon"
                shape["points"] = approx_hand.tolist()
                shape_list.append(shape)

        return shape_list


def run(input_folder):
    image_folder = os.path.join(input_folder, 'image')
    label_folder = os.path.join(input_folder, 'label')
    json_folder = os.path.join(input_folder, 'json')
    if not os.path.exists(json_folder):
        os.mkdir(json_folder)
    assert os.path.exists(image_folder)
    assert os.path.exists(label_folder)

    image_name_list = os.listdir(image_folder)
    label_name_list = os.listdir(label_folder)

    log_list = list()
    name_list = list()
    for label_name in label_name_list:
        if label_name not in image_name_list:
            log_list.append(os.path.join(image_folder, label_name))
        else:
            name_list.append(label_name)

    create_json = CreateJson()
    create_json.to_json(name_list, input_folder)

    if len(log_list):
        print("\nthe following {} image/images not exist!".format(len(log_list)))
        print("{}".format(np.array(log_list).reshape(-1, 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="Path to input images,which has two folders,image and label",
                        required=True, type=str)
    args = parser.parse_args()
    run(args.input_folder)
    time.sleep(1)
    print('\nDemo end!')
