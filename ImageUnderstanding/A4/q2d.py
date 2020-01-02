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
cols, rows= im3_dis.shape
im_dis = [im1_dis, im2_dis, im3_dis]

f = 721.537700
b = 0.5327119288 * 1000
px = 609.559300
py = 172.854000


# # fig = plt.figure()


for n in range(3):
    depth = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            depth[j,i] = (f-b)/im_dis[n][j,i]
            if (im_dis[n][j,i] == 0):
                depth[j,i] = 0
    
    string = 'im'+str(n+1)+'object.txt'
    file1 = open(string, 'r')
    data = []
    for line in file1:
        string = line.split(':')
        data.append(string)
    file1.close()

    digit = []
    for d in data:
        digit.append( [int(s) for s in d[1].split() if s.isdigit()])
    
    centers = []
    for obj in digit:
        xcent = round((obj[0] + obj[2])/2)
        ycent = round((obj[1] + obj[3])/2)
        z = depth[ycent, xcent]
        x = (xcent - px) * z / f
        y = (ycent - py) * z / f
        com = (x,y,z)
        centers.append(com)

    string = 'im'+str(n+1)+'com.txt'
    file2 = open(string, 'w')
    for cord in centers:
        for i in cord:
            file2.write(str(i))
            file2.write(' ')
        file2.write('\n')
    file2.close()
