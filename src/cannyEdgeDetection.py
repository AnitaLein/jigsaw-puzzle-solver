import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def median_filter(image, w):
    """Applies median filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zer-padded to preserve the original resolution.
    """
    height, width, chs = image.shape
    
    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w,w), (w,w), (0,0)))
    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            result[i, j, :] = np.median(image_padded[i:i+2*w+1, j:j+2*w+1, :], axis=(0,1))
    return result
# Canny Edge detection
img = cv.imread('../data/10.jpg')         
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(img,100,200)

# Watershed
se=cv.getStructuringElement(cv.MORPH_RECT , (8,8))
bg=cv.morphologyEx(gray, cv.MORPH_DILATE, se)
out_gray=cv.divide(gray, bg, scale=255)
out_binary=cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU )[1] 

# Denoising?
out_binary=median_filter(out_binary, 3)


 
cv.imshow("Edges", edges)
cv.imshow("gray", out_gray)
cv.imshow("binary", out_binary)
cv.waitKey(0)
cv.destroyAllWindows()