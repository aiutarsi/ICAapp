# Generate mix color png file from more than 2 color png file
import numpy as np
import numpy.linalg as LA
from PIL import Image

def three_to_one(im, height, width, color):
    array = []
    for i in range(height):
        for j in range(width):
            for k in range(color):
                array.append(im[i][j][k])
    return array

def one_to_three(array, height, width, color):
    img = np.zeros((height,width,color))
    count = 0
    for i in range(height):
        for j in range(width):
            for k in range(color):
                img[i][j][k] = array[count]
                count += 1
    return img

### set file names which are read ###
im1o = Image.open('./sample/data1.png')
im2o = Image.open('./sample/data2.png')

im1r = im1o.resize((256, 256))
im2r = im2o.resize((256, 256))
im1r.save('./sample/color1.png')
im2r.save('./sample/color2.png')
im1 = np.array(im1r)
im2 = np.array(im2r)
height = 0
width = 0
color = 0
height = im1.shape[0]
width = im1.shape[1]
color = im1.shape[2]
data1 = np.array(three_to_one(im1,height,width,color))
data2 = np.array(three_to_one(im2,height,width,color))
data = np.zeros((2, len(data1)))
for i in range(2):
    for j in range(len(data1)):
        if (i == 0):
            data[i][j] = data1[j]
        else:
            data[i][j] = data2[j]

### Set Rations in mixing png here. 3 png files -> 3 * 3 Matrix, N png files -> N * N Matrix. pleause select values more than 0 and less than 1###
w = np.array([[0.6,0.4],[0.35,0.65]])
Y = np.dot(w,data)

max_y1 = np.amax(Y[0])
max_y2 = np.amax(Y[1])
min_y1 = np.amin(Y[0])
min_y2 = np.amin(Y[1])
min_y1_abs = abs(min_y1)
min_y2_abs = abs(min_y2)

Y[0] = Y[0] + min_y1_abs
Y[1] = Y[1] + min_y2_abs
Y[0] = Y[0]/(max_y1+min_y1_abs)
Y[1] = Y[1]/(max_y2+min_y2_abs)
Y[0] = 255*Y[0]
Y[1] = 255*Y[1]

im1_new = one_to_three(Y[0],height,width,color)
im2_new = one_to_three(Y[1],height,width,color)

img1 = Image.fromarray(im1_new.astype(np.uint8))
img2 = Image.fromarray(im2_new.astype(np.uint8))
s1 = './sample/mix_color1.png'
s2 = './sample/mix_color2.png'
img1.save(s1)
img2.save(s2)
print("Finished Mixing")
