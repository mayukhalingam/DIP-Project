import numpy as np
import cv2
import matplotlib.pyplot as plt

# reading the input image
img_input = cv2.imread("test_image.jpg",0)
plt.imshow(img_input ,'gray')

##### DENOISING ##############################################
# ret,img = cv2.threshold(op,140,255,cv2.THRESH_BINARY)
ret,img_thre = cv2.threshold(img_input,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  ## otsu's thresholding 
#img_thre = 255 - img_thre
plt.imshow(img_thre,'gray')
plt.show()

img_denoise = cv2.medianBlur(img_thre,3) ## median filter application
plt.imshow(img_denoise,'gray')
plt.show()

### closing :
kernel = np.ones((3,3),np.uint8)
#img_open =  cv2.morphologyEx(img_denoise, cv2.MORPH_OPEN, kernel) 
img_close = cv2.morphologyEx(img_denoise, cv2.MORPH_CLOSE, kernel)
img = cv2.medianBlur(img_close,3)


plt.imshow(img,'gray')
plt.title("denoised image")
plt.show()

