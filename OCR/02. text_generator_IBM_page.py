#!/usr/bin/env python
import numpy as np
import argparse
import glob
import io
import os
import random
import math
import cv2
import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.python.autograph.pyct import origin_info


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  'C:/OCR/text_generator/word_kr.txt')

DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'C:/OCR/text_generator/fonts/')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'F:/CRNN_dataset_kr/hangul-images/')

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 1

# Width and height of the resulting image.
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 768

#============================================
# 노이즈 추가
def salt_and_paper(image, p) :
    output = np.zeros(image.shape,np.uint8)
    there = 1 - p
    for i in range(image.shape[0]) :
        for j in range(image.shape[1]) :
            rdn = random()
            if rdn < p :
                output[i][j] = 0
            elif rdn > there : 
                output[i][j] = 255
            else : 
                output[i][j] = image[i][j]
    return output
#============================================

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

#한글 위치 지정 및 text size 지정
def generate_hangul_location(start, end, text_size):
    while True:
        text_size1 = text_size
        text_size2 = text_size
        text_size3 = text_size
        text_size4 = text_size
        text_size5 = text_size
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
        return x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, math.ceil(text_size1), math.ceil(text_size2), math.ceil(text_size3), math.ceil(text_size4), math.ceil(text_size5)



def generate_hangul_images(label_file, fonts_dir, output_dir):
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()
        print(labels)

    # 폴더 위치
    image_dir = os.path.join(output_dir, 'hangul-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # fonts 가져오기
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    # labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
    #                      encoding='utf-8')

    total_count = 0
    prev_count = 0
    for character in labels:
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))
        
        for font in fonts:
            total_count += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
            font = ImageFont.truetype(font, random.randint(18, 36))
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)
            drawing.text(
                ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                character,
                fill=(0),
                font=font
            )
            file_string = '{}.png'.format(character)
            file_path = os.path.join(image_dir, file_string)
            image.save(file_path, 'png')
            #labels_csv.write(u'{},{}\n'.format(file_path, character))

            for i in range(DISTORTION_COUNT):
                total_count += 1
                file_string = '{}.png'.format(character)
                file_path = os.path.join(image_dir, file_string)
                arr = numpy.array(image)

    print('Finished generating {} images.'.format(total_count))
    #labels_csv.close()

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


    '''
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
'''