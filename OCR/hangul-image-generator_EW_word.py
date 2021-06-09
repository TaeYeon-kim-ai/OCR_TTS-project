#!/usr/bin/env python

import argparse
import glob
import io
import os
import random
import math

import numpy
import pandas as pd
import cv2
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from xml.etree.ElementTree import Element, SubElement, ElementTree


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
# '../labels/2350-common-hangul.txt'
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, 'C:/final_project/IBM/labels/word_kr_208255.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'C:/final_project/IBM/fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'F:/IBM_text_datset/images/') # 'F:/Team Project/OCR/02_Image_to_Text_model/test_data')

TRAIN_ANNOTATION_PATH = 'F:/IBM_text_datset/train_annotation/' # 'F:/Team Project/OCR/02_Image_to_Text_model/test_data/train_annotation/'
TEST_ANNOTATION_PATH = 'F:/IBM_text_datset/test_annotation/' # 'F:/Team Project/OCR/02_Image_to_Text_model/test_data/test_annotation/'
# C:\Users\Admin\Desktop\image-data
# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 1

# Width and height of the resulting image.
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
DEPTH = 5

# Noise Image
noise1 = Image.open('C:/final_project/FRCNN_HEW/noise/noise1.png')
noise2 = Image.open('C:/final_project/FRCNN_HEW/noise//noise2.png')
noise3 = Image.open('C:/final_project/FRCNN_HEW/noise//noise3.png')
noise4 = Image.open('C:/final_project/FRCNN_HEW/noise//noise4.png')
noise5 = Image.open('C:/final_project/FRCNN_HEW/noise//noise5.png')
noise6 = Image.open('C:/final_project/FRCNN_HEW/noise//noise6.png')
noise1 = noise1.resize((int(noise1.width / 2), int(noise1.height / 2)))
noise2 = noise2.resize((int(noise2.width / 2), int(noise2.height / 2)))
noise3 = noise3.resize((int(noise3.width / 2), int(noise3.height / 2)))
noise4 = noise4.resize((int(noise4.width / 2), int(noise4.height / 2)))
noise5 = noise5.resize((int(noise5.width / 2), int(noise5.height / 2)))
noise6 = noise6.resize((int(noise6.width / 2), int(noise6.height / 2)))

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 1
        sigma = var**0.5
        gauss = numpy.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 1
        amount = 0.004
        out = numpy.copy(image)
        # Salt mode
        num_salt = numpy.ceil(amount * image.size * s_vs_p)
        coords = [numpy.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = numpy.ceil(amount* image.size * (1. - s_vs_p))
        coords = [numpy.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(numpy.unique(image))
        vals = 2 ** numpy.ceil(numpy.log2(vals))
        noisy = numpy.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = numpy.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def generate_hangul_location(start, end, text_size):
    while True:
        text_size1 = text_size/3
        text_size2 = text_size/2
        text_size3 = text_size
        text_size4 = text_size/2
        text_size5 = text_size/2
        x1 = random.randint(start + math.ceil(text_size1), end - math.ceil(text_size1))
        y1 = random.randint(start + math.ceil(text_size1), end - math.ceil(text_size1))

        x2 = random.randint(start + math.ceil(text_size2), end - math.ceil(text_size2))
        y2 = random.randint(start + math.ceil(text_size2), end - math.ceil(text_size2))

        x3 = random.randint(start + math.ceil(text_size3), end - math.ceil(text_size3))
        y3 = random.randint(start + math.ceil(text_size3), end - math.ceil(text_size3))

        x4 = random.randint(start + math.ceil(text_size4), end - math.ceil(text_size4))
        y4 = random.randint(start + math.ceil(text_size4), end - math.ceil(text_size4))

        x5 = random.randint(start + math.ceil(text_size5), end - math.ceil(text_size5))
        y5 = random.randint(start + math.ceil(text_size5), end - math.ceil(text_size5))

        # dis1 = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # dis2 = math.sqrt((x1 - x3)**2 + (y1 - y3)**2)
        # dis3 = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
        # dis4 = math.sqrt((x1 - x4)**2 + (y1 - y4)**2)
        # dis5 = math.sqrt((x2 - x4)**2 + (y2 - y4)**2)
        # dis6 = math.sqrt((x3 - x4)**2 + (y3 - y4)**2)
        # dis7 = math.sqrt((x1 - x5)**2 + (y1 - y5)**2)
        # dis8 = math.sqrt((x2 - x5)**2 + (y2 - y5)**2)
        # dis9 = math.sqrt((x3 - x5)**2 + (y3 - y5)**2)
        # dis10 = math.sqrt((x4 - x5)**2 + (y4 - y5)**2)
        

        # if dis1 < (text_size1 + text_size2)/2 or dis2 < (text_size1 + text_size3)/2 or dis3 < (text_size2 + text_size3)/2 or dis4 < (text_size1 + text_size4)/2 or dis5 < (text_size2 + text_size4)/2 or dis6 < (text_size3 + text_size4)/2 or dis7 < (text_size1 + text_size5)/2 or dis8 < (text_size2 + text_size5)/2 or dis9 < (text_size3 + text_size5)/2 or dis10 < (text_size4 + text_size5)/2:
        #     continue
        # elif x1 > 280 or x2> 280 or x3> 280 or x4 > 280 or x5 > 280 :
        #     continue
        # else:
        return x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, math.ceil(text_size1), math.ceil(text_size2), math.ceil(text_size3), math.ceil(text_size4), math.ceil(text_size5)

def generate_image_noise(img):
    img.paste(noise1, (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)))
    img.paste(noise2, (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)))
    img.paste(noise3, (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)))
    img.paste(noise4, (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)))
    img.paste(noise5, (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)))
    img.paste(noise6, (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)))

    return img

def generate_annotation_xml(filename, path, width, height, depth, text_size1, text_size2, text_size3, text_size4, text_size5, name, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, word_len):
    root = Element('annotation')
    SubElement(root, 'folder').text = 'images'

    SubElement(root, 'filename').text = filename + '.png'

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = str(depth)

    SubElement(root, 'segmented').text = '0'

    # object 1
    object = SubElement(root, 'object')
    SubElement(object, 'name').text = "text"
    SubElement(object, 'pose').text = 'Unspecified'
    SubElement(object, 'truncated').text = '0'
    SubElement(object, 'occluded').text = '0'
    SubElement(object, 'difficult').text = '0'

    bnd = SubElement(object, 'bndbox')
    # SubElement(bnd, 'xmin').text = str(x1 - text_size1/2)
    # SubElement(bnd, 'ymin').text = str(y1 - text_size1/2)
    # SubElement(bnd, 'xmax').text = str(x1 + text_size1/2)
    # SubElement(bnd, 'ymax').text = str(y1 + text_size1/2)
    SubElement(bnd, 'xmin').text = str(x1)
    SubElement(bnd, 'ymin').text = str(y1)
    SubElement(bnd, 'xmax').text = str(x1 + text_size1*(word_len-3))
    SubElement(bnd, 'ymax').text = str(y1 + text_size1)


    # object 2
    object = SubElement(root, 'object')
    SubElement(object, 'name').text = "text"
    SubElement(object, 'pose').text = 'Unspecified'
    SubElement(object, 'truncated').text = '0'
    SubElement(object, 'occluded').text = '0' 
    SubElement(object, 'difficult').text = '0'

    bnd = SubElement(object, 'bndbox')
    # SubElement(bnd, 'xmin').text = str(x2 - text_size2/2)
    # SubElement(bnd, 'ymin').text = str(y2 - text_size2/2)
    # SubElement(bnd, 'xmax').text = str(x2 + text_size2/2)
    # SubElement(bnd, 'ymax').text = str(y2 + text_size2/2)
    SubElement(bnd, 'xmin').text = str(x2)
    SubElement(bnd, 'ymin').text = str(y2)
    SubElement(bnd, 'xmax').text = str(x2 + text_size2*(word_len-3))
    SubElement(bnd, 'ymax').text = str(y2 + text_size2)


    # object 3
    object = SubElement(root, 'object')
    SubElement(object, 'name').text = "text"
    SubElement(object, 'pose').text = 'Unspecified'
    SubElement(object, 'truncated').text = '0'
    SubElement(object, 'occluded').text = '0'
    SubElement(object, 'difficult').text = '0'

    bnd = SubElement(object, 'bndbox')
    # SubElement(bnd, 'xmin').text = str(x3 - text_size3/2)
    # SubElement(bnd, 'ymin').text = str(y3 - text_size3/2)
    # SubElement(bnd, 'xmax').text = str(x3 + text_size3/2)
    # SubElement(bnd, 'ymax').text = str(y3 + text_size3/2)
    SubElement(bnd, 'xmin').text = str(x3)
    SubElement(bnd, 'ymin').text = str(y3)
    SubElement(bnd, 'xmax').text = str(x3 + text_size3*(word_len-3))
    SubElement(bnd, 'ymax').text = str(y3 + text_size3)

        # object 4
    object = SubElement(root, 'object')
    SubElement(object, 'name').text = "text"
    SubElement(object, 'pose').text = 'Unspecified'
    SubElement(object, 'truncated').text = '0'
    SubElement(object, 'occluded').text = '0'
    SubElement(object, 'difficult').text = '0'

    bnd = SubElement(object, 'bndbox')
    # SubElement(bnd, 'xmin').text = str(x3 - text_size3/2)
    # SubElement(bnd, 'ymin').text = str(y3 - text_size3/2)
    # SubElement(bnd, 'xmax').text = str(x3 + text_size3/2)
    # SubElement(bnd, 'ymax').text = str(y3 + text_size3/2)
    SubElement(bnd, 'xmin').text = str(x4)
    SubElement(bnd, 'ymin').text = str(y4)
    SubElement(bnd, 'xmax').text = str(x4 + text_size4*(word_len-3))
    SubElement(bnd, 'ymax').text = str(y4 + text_size4)

        # object 5
    object = SubElement(root, 'object')
    SubElement(object, 'name').text = "text"
    SubElement(object, 'pose').text = 'Unspecified'
    SubElement(object, 'truncated').text = '0'
    SubElement(object, 'occluded').text = '0'
    SubElement(object, 'difficult').text = '0'

    bnd = SubElement(object, 'bndbox')
    # SubElement(bnd, 'xmin').text = str(x3 - text_size3/2)
    # SubElement(bnd, 'ymin').text = str(y3 - text_size3/2)
    # SubElement(bnd, 'xmax').text = str(x3 + text_size3/2)
    # SubElement(bnd, 'ymax').text = str(y3 + text_size3/2)
    SubElement(bnd, 'xmin').text = str(x5)
    SubElement(bnd, 'ymin').text = str(y5)
    SubElement(bnd, 'xmax').text = str(x5 + text_size5*(word_len-3))
    SubElement(bnd, 'ymax').text = str(y5 + text_size5)



    tree = ElementTree(root)
    tree.write(path + filename + '.xml')

#====================================================================










#====================================================================
def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    """
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    # image_dir = os.path.join(output_dir, 'hangul-images')

    TRAIN_IMAGE_DIR = os.path.join(output_dir, 'train_hangul-images')
    TEST_IMAGE_DIR = os.path.join(output_dir, 'test_hangul-images')
    if not os.path.exists(TRAIN_IMAGE_DIR):
        os.makedirs(os.path.join(TRAIN_IMAGE_DIR))
    if not os.path.exists(TEST_IMAGE_DIR):
        os.makedirs(os.path.join(TEST_IMAGE_DIR))
    # if not os.path.exists(image_dir):
    #     os.makedirs(os.path.join(image_dir))
    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    # labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
    #                      encoding='utf-8')

    test_total_count = 0
    train_total_count = 0
    prev_count = 0
    text_size = 20
    # TRAIN_ANNOTATION_PATH = 'F:/Team Project/OCR/01_Text_detection/data/train_annotation/' # 'F:/Team Project/OCR/02_Image_to_Text_model/test_data/train_annotation/'
    # TEST_ANNOTATION_PATH = 'F:/Team Project/OCR/01_Text_detection/data/test_annotation/' # 'F:/Team Project/OCR/02_Image_to_Text_model/test_data/test_annotation/'
    if not os.path.exists(TRAIN_ANNOTATION_PATH):
        os.makedirs(os.path.join(TRAIN_ANNOTATION_PATH))
    if not os.path.exists(TEST_ANNOTATION_PATH):
        os.makedirs(os.path.join(TEST_ANNOTATION_PATH))

    label_list = []
    path_list = []
    for character in labels:
        # Print image count roughly every 5000 images.
        if test_total_count - prev_count > 10000:
            prev_count = test_total_count
            print('{} test images generated...'.format(test_total_count))

        for font in fonts:
            
            # 위치 생성
            x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, text_size1, text_size2, text_size3, text_size4, text_size5 = generate_hangul_location(0, 500, text_size)
            word_len = len(character)
            test_total_count += 1
            image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=(255, 255, 255)) # 'L' -> 'RGB'
            # font = ImageFont.truetype(font, text_size)
            font1 = ImageFont.truetype(font, text_size1)
            font2 = ImageFont.truetype(font, text_size2)
            font3 = ImageFont.truetype(font, text_size3)
            font4 = ImageFont.truetype(font, text_size4)
            font5 = ImageFont.truetype(font, text_size5)
            drawing = ImageDraw.Draw(image)
            # w, h = drawing.textsize(character, font=font)
            # w1, h1 = drawing.textsize(character, font = font1)
            # w2, h2 = drawing.textsize(character, font = font2)
            # w3, h3 = drawing.textsize(character, font = font3)

            # drawing 1
            drawing.text(
                (x1, y1), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0, 0, 0),
                font=font1
            )
            # drawing 2
            drawing.text(
                (x2, y2), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0, 0, 0),
                font=font2
            )
            # drawing 3
            drawing.text(
                (x3, y3), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0, 0, 0),
                font=font3
            )

                        
            # drawing 4
            drawing.text(
                (x4, y4), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0, 0, 0),
                font=font4
            )

            # drawing 5
            drawing.text(
                (x5, y5), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0, 0, 0),
                font=font5
            )

            file_string = 'hangul_{}.png'.format(test_total_count)
            file_path = os.path.join(TEST_IMAGE_DIR, file_string)
            # shape = numpy.array(image)
            # print('shape.shape: ', shape.shape)
            image.save(file_path, 'PNG')

            annotation_name1 = 'hangul_{}'.format(test_total_count)
            generate_annotation_xml(annotation_name1, TEST_ANNOTATION_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH,
                                    text_size1, text_size2, text_size3,text_size4, text_size5, character, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, word_len)

            # label_list.append(character)
            # path_list.append(file_path)

            # labels_csv.write(u'{},{}\n'.format(file_path, character))

            for i in range(DISTORTION_COUNT):
                train_total_count += 1
                file_string = 'hangul_{}.png'.format(train_total_count)
                file_path = os.path.join(TRAIN_IMAGE_DIR, file_string)
                ######################################################################
                # 노이즈 background 생성
                image2 = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=(255, 255, 255)) # 'L' -> 'RGB'
                image2 = Image.open('C:/final_project/FRCNN_HEW/noise/paper_texture.png')
                image2 = image2.resize((512, 512))

                image2 = generate_image_noise(image2)
                ######################################################################
                # 글자 위치 생성
                x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, text_size1, text_size2, text_size3, text_size4, text_size5 = generate_hangul_location(0, 500, text_size)

                font1 = ImageFont.truetype(font, text_size1)
                font2 = ImageFont.truetype(font, text_size2)
                font3 = ImageFont.truetype(font, text_size3)
                font4 = ImageFont.truetype(font, text_size4)
                font5 = ImageFont.truetype(font, text_size5)
                drawing = ImageDraw.Draw(image2)

                # drawing 1
                drawing.text(
                    (x1, y1), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                    character,
                    fill=(0, 0, 0),
                    font=font1
                )
                # drawing 2
                drawing.text(
                    (x2, y2), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                    character,
                    fill=(0, 0, 0),
                    font=font2
                )
                # drawing 3
                drawing.text(
                    (x3, y3), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                    character,
                    fill=(0, 0, 0),
                    font=font3
                )

                # drawing 4
                drawing.text(
                    (x4, y4), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                    character,
                    fill=(0, 0, 0),
                    font=font4
                )

                # drawing 5
                drawing.text(
                    (x5, y5), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                    character,
                    fill=(0, 0, 0),
                    font=font5
                )
                ######################################################################
                image2 = numpy.asarray(image2)
                noise_image = noisy('s&p', image2)        # gauss s&p poisson speckle

                noise_image = Image.fromarray(noise_image.astype('uint8'))

                noise_image.save(file_path, 'PNG')
                annotation_name2 = 'hangul_{}'.format(train_total_count)
                generate_annotation_xml(annotation_name2, TRAIN_ANNOTATION_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH,
                                        text_size1, text_size2, text_size3, text_size4, text_size5, character, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, word_len)
                # label_list.append(character)
                # path_list.append(file_path)

                # labels_csv.write(u'{},{}\n'.format(file_path, character))

    # label_list = numpy.array(label_list)
    # path_list = numpy.array(path_list)

    # label_list = label_list.reshape(-1, 1)
    # path_list = path_list.reshape(-1, 1)

    # final_list = numpy.concatenate((path_list, label_list), axis = 1)

    # final_list = pd.DataFrame(final_list)
    # final_list.to_csv('F:/Team Project/OCR/02_Image_to_Text_model/test_data/test-labels-map.csv'
    #                   , index_label = False
    #                   , header = False)


    print('Finished generating {} train images.'.format(train_total_count))
    print('Finished generating {} test images.'.format(test_total_count))
    # labels_csv.close()


def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.

    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = numpy.random.RandomState(None)
    shape = image.shape
    print('image.shape: ', image.shape)
    print('shape: ', shape)

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)

# DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
#                                   '../labels/2350-common-hangul.txt')
# DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../fonts')
# DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'C:/Users/Admin/Desktop/image-data')
# # C:\Users\Admin\Desktop\image-data
# # Number of random distortion images to generate per font and character.
# DISTORTION_COUNT = 3

# # Width and height of the resulting image.
# IMAGE_WIDTH = 64
# IMAGE_HEIGHT = 64