import cv2
import matplotlib.pyplot as plt
def cv2_show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
def cv2_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

import numpy as np

import os
os.environ['PATH']+=":/Volumes/data/tesseract/bin"
print(os.environ['PATH'])
os.system("tesseract help")

from PIL import Image
import pytesseract

image = cv2.imread('images/scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#preprocess = 'thresh'
preprocess = 'median'
if preprocess == 'thresh':
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
else:
    gray = cv2.medianBlur(gray, 3)
cv2_show(gray, 'gray_'+preprocess)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

img = Image.open(filename)

text = pytesseract.image_to_string(img)

print(text)