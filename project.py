import numpy as np
import cv2
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import argparse
import os
from skimage.feature import hog
from scipy import ndimage
from skimage.restoration import denoise_nl_means

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
	#cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


def encode(ip):
    if(ip<6):
        op=str(0)+str(91)+str(ip+4)
    if(ip>=6 and ip<=11):
        temp=ip+4
        if(temp==10):
            op=str(0)+str(91)+'A'
        if(temp==11):
            op=str(0)+str(91)+'B'
        if(temp==12):
            op=str(0)+str(91)+'C'
        if(temp==13):
            op=str(0)+str(91)+'D'
        if(temp==14):
            op=str(0)+str(91)+'E'
        if(temp==15):
            op=str(0)+str(91)+'F'

    if(ip>11 and ip<=26):
        temp=ip-11
        if(temp<10):
            op=str(0)+str(92)+str(temp)
        if(temp==10):
            op=str(0)+str(92)+'A'
        if(temp==11):
            op=str(0)+str(92)+'B'
        if(temp==12):
            op=str(0)+str(92)+'C'
        if(temp==13):
            op=str(0)+str(92)+'D'
        if(temp==14):
            op=str(0)+str(92)+'E'
        if(temp==15):
            op=str(0)+str(92)+'F'
    if(ip==27):
        op=str(0)+str(930)
    if(ip==28):
        op=str(0)+str(932)
    if(ip>28):
        op=str(0)+str(93)+str(ip-24)
    
    return op

def splitting(img):
    l,w = img.shape[0:2]
    l1 = int(l/2)
    w1 = int(w/2)
    if w%2 == 0:
        w=w-1
    if l%2 == 0:
        l=l-1
    if w1%2 == 0:
        w1=w1+1
    if l1%2 == 0:
        l1=l1+1
    
    print(w,l,l1,w1)
    s = []
    print("shapessss")
    s.append(img[0:l1,0:w1])
    s.append(img[0:l1,-w1:])
    s.append(img[-l1:,0:w1])
    s.append(img[-l1:,-w1:])
    s.append(img[0:l1,0:w])
    s.append(img[-l1:,0:w])
    s.append(img[0:l,0:w1])
    s.append(img[0:l,-w1:])
    
    return s

def correlation(img,x,y,s1):
    ans=[]
    s2=extract_template(img,x,y)
    
    for i in range(8):
        Is=s2[i]
        Itp=s1[i]
        Is_bar=np.ones(Is.shape)*np.mean(Is)
        Itp_bar=np.ones(Itp.shape)*np.mean(Itp)
        Is_diff=Is-Is_bar
        Itp_diff=Itp-Itp_bar
        
        ans.append(np.sum((Is_diff)*(Itp_diff))/np.sqrt(np.sum((Is_diff**2))*np.sum((Itp_diff**2))))
    
    ans=np.asarray(ans)
    ans=np.nan_to_num(ans)
        
    return ans




#done = True
done=False

while(done==False):
    
    ref_point = [] 
    crop = False

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
        [sg.Text('Type yes if all the characters are marked', size=(5, 1)), sg.InputText()],
       
        [sg.Submit(), sg.Cancel()]
    ]

    window = sg.Window('ID-Window', layout)
    event, values = window.Read()
    window.Close()

    


    ip=values[0]
    ip=int(ip)
    op_code = encode(ip)
    print(op_code)
    
    if(values[1]=='yes'):
        done=True
    
    l=int(crop_img.shape[0]/2)
    w=int(crop_img.shape[1]/2)
    
    templates = splitting(crop_img)
    mat = []
    for template in templates:
        t_w = int(template.shape[0]/2)
        t_h = int(template.shape[1]/2)
        im_pad = np.pad(img,((t_w,t_w),(t_h,t_h)),'constant')
        ret=cv2.matchTemplate(im_pad,template,cv2.TM_CCORR_NORMED)
        filtered = ndimage.maximum_filter(ret, size=4)
        mat.append(filtered)

    s = np.array(mat[0])
    for i in range(1,len(mat)):
        s = s+np.array(mat[i])

    op=np.zeros(ret.shape)
    
    ret=cv2.matchTemplate(img,crop_img,cv2.TM_CCORR_NORMED)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if(ret[i][j]>0.75):
                op[i][j]=255

    plt.imshow(op,'gray')
    plt.show()
    
    index=np.where(op==255)
    index=np.asarray(index)
    index.sort()


    t=10
    a=[]
    i=0
    
    fd_img, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(4, 4), visualize=True, multichannel=False)
    fd_crop, hog_temp = hog(crop_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(4, 4), visualize=True, multichannel=False)

    while(i<index.shape[1]):
        print(i)
        mx=index[0][i]
        my=index[1][i]
        count=1
        for j in range(i+1,index.shape[1]):
            marginx=np.abs(index[0][i]-index[0][j])
            marginy=np.abs(index[1][i]-index[1][j])
            if(marginx<t and marginy<t):
                mx=mx+index[0][j]
                my=my+index[1][j]
                count=count+1
                step=j
        print(count,step)
        a.append([mx//count,my//count])
        if(count>1):
            i=step+1
        else:
            i=i+1

    a=np.asarray(a)
    print(a.shape)
    
    match_scores=[]
    patches=[]
    match_coord = []
    print(a.shape[0])
    for i in range(a.shape[0]):
        r,w = crop_img.shape
        l1=int(r/2)
        w1=int(w/2)
        x=a[i][0]
        y=a[i][1]
        print(r,w)
        print(l1,w1)

        lr = x
        hr = x+r
        lw = y
        hw = y+w

        patch = img[lr:hr,lw:hw]
        
        fd_patch, hog_patch = hog(patch, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(4, 4), visualize=True, multichannel=False)
        norm1=np.linalg.norm(fd_patch,2)
        norm2=np.linalg.norm(fd_crop,2)
        product = np.multiply(fd_patch,fd_crop).sum()/(norm1*norm2)
    #     print(product) 
        if product >= 0.5:
            match_scores.append(product)
            patches.append(patch)
            match_coord.append([lr,hr,lw,hw])

    print(match_scores)
    match_scores = np.array(match_scores)
    sort = match_scores.argsort()
    print("sorted",sort)
    for i in range(len(sort)):
        print(match_scores[sort[i]])
        plt.imshow(patches[sort[i]],'gray')
        plt.show()
        
    #patches=np.asarray(patches)
    for i in range(len(patches)):
        img[match_coord[i][0]:match_coord[i][1], match_coord[i][2]:match_coord[i][3]] = 0
        


############################## file generation #############################

a=np.array(a)
f = open('1.txt', 'a')
for i in range(a.shape[0]):
    f.write(str(a[i][0]))
    f.write(' '+str(a[i][1]))
    f.write(' '+op_code+'\n')

f.close()
############################### editable text generation ###################
Info1 = open('1.txt','r')
c = 0
location_matrix = []
location_matrix2 = []
for x in Info1:
    p = x.split()
    c1 = 0
    c += 1
    for i in p:
        c1 += 1
        if c1 < 3:
            g = int(i)
            location_matrix.append(g)
        else:
            location_matrix2.append(i)

location_matrix = np.reshape(location_matrix,(c,2)).astype(int)
location_matrix2 = np.asarray(location_matrix2)
## sorting
for i in range(1,c):
    temp = np.copy(location_matrix[i])
    j = i - 1
    while j >= 0 and temp[0] < location_matrix[j][0]:
        location_matrix[j+1] = np.copy(location_matrix[j])
        j = j - 1
    location_matrix[j+1] = np.copy(temp)
for i  in range(1,c):
    if(abs(location_matrix[i][0] - location_matrix[i-1][0]) <= 25):
        location_matrix[i][0] = np.copy(location_matrix[i-1][0])
print(location_matrix)
for i in range(1,c):
    temp = np.copy(location_matrix[i])
    j = i - 1
    while j >= 0 and temp[1] < location_matrix[j][1] and location_matrix[j][0] == temp[0]:
        location_matrix[j+1][1] = np.copy(location_matrix[j][1])
        j =  j - 1
    location_matrix[j+1][1] = np.copy(temp[1])
print(location_matrix)
rows = np.unique(location_matrix[:,0],False,False,False,None)
print(rows)
final = np.vstack((location_matrix[:,0],location_matrix[:,1],location_matrix2))
final1 = np.transpose(final)
print(final1)

for i in range(c):
    x = final1[i][2]
    y =  '\\u'+ str(x)
    space = " "
    y = y.strip()
    space = space.encode("utf-8")
    y = y.encode("utf-8")
    y = y.decode('unicode_escape')
    space = space.decode('unicode_escape')
    final.write((y.encode('utf8')))
    final.write(space.encode('utf8'))

    if int(final1[i][1]) >= img.shape[1]-120:
        print("y")
        nex = "\n"
        nex = nex.encode('utf-8')
        nex = nex.decode('unicode_escape')
        final.write(nex.encode('utf8'))

final.close()



