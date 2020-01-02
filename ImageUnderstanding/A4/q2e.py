import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math

im1 = '004945'
im2 = '004964'
im3 = '005002'
results = 'data/test/results/'
left_disparity = '_left_disparity.png'
calib = 'data/test/calib'
allcalib = '_allcalib.txt'


im1_left = results + im1 + left_disparity
im2_left = results + im2 + left_disparity
im3_left = results + im3 + left_disparity


im1_dis = cv2.imread(im1_left, 0)
im2_dis = cv2.imread(im2_left, 0)
im3_dis = cv2.imread(im3_left, 0)
im1 = 'data/test/left.004945.jpg'
im2 = 'data/test/left.004964.jpg'
im3 = 'data/test/left.005002.jpg'
im3 = cv2.imread(im3, 0)
cols, rows= im3_dis.shape
im_dis = [im1_dis, im2_dis, im3_dis]

f = 721.537700
b = 0.5327119288 * 1000
px = 609.559300
py = 172.854000

fig = plt.figure()
for n in range(3):
    depth = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            depth[j,i] = (f-b)/im_dis[n][j,i]
            if (im_dis[n][j,i] == 0):
                depth[j,i] = 0

    segmentation = np.zeros((cols, rows))

    string = 'im' + str(n+1) + 'com.txt'
    file1 = open(string, 'r')

    com = []
    for line in file1:
        com.append ([float(s) for s in line.split()])

    file1.close()

    string = 'im' + str(n+1) + 'object.txt'
    file2 = open(string, 'r')
    data = []
    for line in file2:
        data.append([int(s) for s in line.split() if s.isdigit()])
    

    for obj in data:
        c = com[data.index(obj)]
        for i in range(obj[0], obj[2], 1):
            for j  in range(obj[1], obj[3], 1):
                z = depth[min(j,cols-1), min(i,rows-1)]
                x = (i - px)*z/f
                y = (j - py)*z/f
                cord = np.array((x,y,z))
                if (np.linalg.norm(c - cord) <= 3):
                    segmentation[min(j,cols-1),min(i,rows-1)] = data.index(obj)+1
    fig.add_subplot(3,1, n+1)

    plt.imshow(segmentation) 
    file2.close()   
plt.show()

            