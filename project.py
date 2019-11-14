import numpy as np
import cv2
import matplotlib.pyplot as plt
import PySimpleGUI as sg

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


################# taking inputs ################################

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables 

    global ref_point, crop

    # if the left mouse button was clicked, record the starting 
    # (x, y) coordinates and indicate that cropping is being performed 
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released 
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that 
        # the cropping operation is finished 
        ref_point.append((x, y))

        # draw a rectangle around the region of interest 
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)




# load the image, clone it, and setup the mouse callback function 
image = img
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)


# keep looping until the 'q' key is pressed 
while True:
        # display the image and wait for a keypress 
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # press 'r' to reset the window 
    if key == ord("r"):
        image = clone.copy()

        # if the 'c' key is pressed, break from the loop 
    elif key == ord("c"):
        break

if len(ref_point) == 2:
    crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:
                                                        ref_point[1][0]]
    a=ref_point[0]
    b=ref_point[1]
    xt=(a[0]+b[0])//2
    yt=(a[1]+b[1])//2
    cv2.imshow("crop_img", crop_img)
    cv2.waitKey(0)


cv2.destroyAllWindows()

layout = [
    [sg.Text('Please enter the character id')],
    [sg.Text('id', size=(15, 1)), sg.InputText()],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('ID-Window', layout)
event, values = window.Read()
window.Close()



