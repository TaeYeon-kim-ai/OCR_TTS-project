#image resize 종횡비 유지
import cv2
# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]
    # Return original image if no need to resize
    if width is None and height is None:
        return image
    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)
if __name__ == '__main__':
    image = cv2.imread('1.png')
    cv2.imshow('image', image)
    resize_ratio = 1.2
    resized = maintain_aspect_ratio_resize(image, width=int(image.shape[1] * resize_ratio))
    cv2.imshow('resized', resized)
    cv2.imwrite('resized.png', resized)
    cv2.waitKey(0)

