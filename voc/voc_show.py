from argparse import ArgumentParser
from glob import glob
import os
import sys
import cv2
from common_function import PascalVocReader, PascalVocVisualize


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="Path to input folder", required=True, type=str)
    return parser

def main():
    args = build_argparser().parse_args()
    input_dir = args.input_dir
    images_path = glob(os.path.join(input_dir, ) + '/*.jpg')
    xmls_path = glob(os.path.join(input_dir, ) + '/*.xml')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    for image_path in images_path:
        print(image_path)
        image = cv2.imread(image_path)
        xml_path = image_path.replace('image', 'xml')
        xml_path = xml_path.replace('jpg', 'xml')

        if xml_path in xmls_path:
            voc_read = PascalVocReader(xml_path)
            voc_shapes = voc_read.getShapes()

            bbox_list = list()
            category_list = list()

            for voc_shape in voc_shapes:
                category_list.append(voc_shape[0])
                bbox_list.append(voc_shape[1])
            annotations = {'image': image,
                           'bboxes': bbox_list,
                           'categories': category_list}

            voc_visualize = PascalVocVisualize(annotations)
            show_image = voc_visualize.get_visualize()

            cv2.imshow("image", show_image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break


if __name__ == '__main__':
    sys.exit(main() or 0)


