import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib.patches import ConnectionPatch
import matplotlib
from scipy.spatial.distance import cdist
from numpy.linalg import inv



def matching(dist,kp, kp2):
    cord = []
    newkp=[]
    newkp2=[]
    threshold = 0.6
    for i in range(len(dist)):
        smallest = float("inf")
        second = float("inf")
        smallest_index = 0
        second_index =0
        for j in range(len(dist[i])):
            value = dist[i][j]

            if (value < second and value != 0):
                if (value < smallest):
                    second = smallest
                    second_index = smallest_index
                    smallest = value
                    smallest_index = j
                else:
                    second = value
                    second_index = j
            # if (smallest == 0 or second == 0):
        if ( smallest/second < threshold):
            newkp.append(kp[i])
            newkp2.append(kp2[smallest_index])
            cord.append((i, smallest_index, smallest))
    
    matches = []   
    for i in range(len(newkp)):
        matches.append(cv2.DMatch(_queryIdx = i, _trainIdx = i, _distance = cord[i][2]))
    temp = []
    for i in range(len(matches)):
        temp.append((newkp[i],newkp2[i],matches[i]))
    return temp


if __name__ == '__main__':

    img1 = cv2.imread('./images/colourTemplate.png')
    img2 = cv2.imread('./images/colourSearch.png')

    



    img1blue = img1.copy()
    img1blue[:,:,1] = 0
    img1blue[:,:,2] = 0
    img1green = img1.copy()
    img1green[:,:,0] = 0
    img1green[:,:,2] = 0
    img1red = img1.copy()
    img1red[:,:,0] = 0
    img1red[:,:,1] = 0

    img2blue = img2.copy()
    img2blue[:,:,1] = 0
    img2blue[:,:,2] = 0
    img2green = img2.copy()
    img2green[:,:,0] = 0
    img2green[:,:,2] = 0
    img2red = img2.copy()
    img2red[:,:,0] = 0
    img2red[:,:,1] = 0

    

    sift1 = cv2.xfeatures2d.SIFT_create(sigma = 2)
    sift2 = cv2.xfeatures2d.SIFT_create( sigma = 1.6)

    kp1b, des1b = sift1.detectAndCompute(img1blue,None)
    kp1g, des1g = sift1.detectAndCompute(img1green,None)
    kp1r, des1r = sift1.detectAndCompute(img1red,None)
    kp2b, des2b = sift2.detectAndCompute(img2blue,None)
    kp2g, des2g = sift2.detectAndCompute(img2green,None)
    kp2r, des2r = sift2.detectAndCompute(img2red,None)

    des1 = [des1b, des1g, des1r]
    des2 = [des2b, des2g, des2r]
    
    matchB = []
    matchG = []
    matchR = []
    
    if (des1[0] is not None and des2[0] is not None):
        distB = cdist(des1[0], des2[0], 'euclidean')
        matchB = matching(distB, kp1b, kp2b)
    if (des1[1] is not None and des2[2] is not None):
        distG = cdist(des1[1], des2[2], 'euclidean')
        matchG = matching(distG, kp1g, kp2g)
    if (des1[2] is not None and des2[2] is not None):
        distR = cdist(des1[2], des2[2], 'euclidean')
        matchR = matching(distR, kp1r, kp2r)

    matches = (matchB + matchG) + matchR
    matches.sort(key=lambda tup: tup[2].distance) 
    k = 5
    best = matches[:k]
    M = np.zeros((2*k, 6))
    M2 = np.zeros((2*k,1))
    A = np.zeros((2*k,1))
    j = 0
    for i in range(len(best)):
        x = best[i][0].pt[0]
        y = best[i][0].pt[1]
        x2 = best[i][1].pt[0]
        y2 = best[i][1].pt[1]   
        M[j,0] = x
        M[j,1] = y
        M[j+1,2] = x
        M[j+1,3] = y
        M[j,4] = 1
        M[j+1,5] = 1
        M2[j] = x2
        M2[j+1] = y2
        j +=2

    np.set_printoptions(suppress=True)
    A = np.dot(np.dot(inv(np.dot( np.transpose(M) , M) ), np.transpose(M)) , M2)


    rows,cols,ch = img1.shape

    P = np.array([[1,1,0,0,1,0],
         [0,0,1,1,0,1],
         [cols,1,0,0,1,0],
         [0,0,cols,1,0,1],
         [1,rows,0,0,1,0],
         [0,0,1,rows,0,1],
         [cols,rows,0,0,1,0],
         [0,0,cols,rows,0,1]])
    
    P = np.dot(P, A)
    
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.plot((P[0,0], P[2,0]), (P[1,0], P[3,0]), 'ro-')
    plt.plot((P[0,0], P[4,0]), (P[1,0], P[5,0]), 'ro-')
    plt.plot((P[6,0], P[4,0]), (P[7,0], P[5,0]), 'ro-')
    plt.plot((P[6,0], P[2,0]), (P[7,0], P[3,0]), 'ro-')
    plt.show()

    # temp = np.zeros((6,2))
    # A = np.c_[A, temp]
    # A[2,2] = 1
    # print A

    # corners = np.ones([6,3])
    # corners[2,0] = rows
    # corners[3,0] = cols
    
    # target = np.dot(corners, A)
    # print target

    # dst = cv2.warpAffine(img1,A,(cols,rows))
    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(dst),plt.title('Output')
    # plt.show()

    # for i in cord:
    #     if (des[i[0]] is not None and des2[i[1]] is not None):
    #         matches.append(cv2.DMatch(_queryIdx = i[1], _trainIdx = i[0], _distance = i[2]))
    #     else:
    #         print ("huh")
    # print matches[0].queryIdx
    # print des2[matches[0].queryIdx]
    # print matches[0].trainIdx
    # print des[matches[0].trainIdx]
    # img3 = cv2.drawMatches(img1,kp,img2,kp2,newMatches,outImg = None)
    



    
    # cord = []
    # newkp = []
    # newkp2= []
    # print("where am i")
    # for i in range(len(kpDes1)):
    #     lmin = float("inf")
    #     hmin = float("inf")
    #     lminCor = 0
    #     hminCor = 0
    #     d1 = np.transpose(kpDes1[i][1])
    #     for j in range(len(kpDes2)):
    #         d2 = np.transpose(kpDes2[j][1])
    #         dist = distance.euclidean( d1, d2)
    #         # dist = distance.euclidean(kpDes1[i][0].pt, kpDes2[j][0].pt)
    #         if (dist < lmin):
    #             hmin = lmin
    #             lmin = dist
    #             hminCor = lminCor
    #             lminCor = j
    #         elif dist < hmin:
    #             hmin = dist
    #             hminCor = j
    #     if (lmin/hmin < 0.8):
    #         cord.append((i, lminCor,dist))
    #         newkp.append(kpDes1[i][0])
    #         newkp2.append(kpDes2[j][0])
    
    # print(len(newkp), len(newkp2))
    # for i in newkp:
    #     print i
    #     print i.pt
    # fig = plt.figure(figsize=(10,5))
    
    
    # img1=cv2.drawKeypoints(img1,newkp, None)
    # img2=cv2.drawKeypoints(img2,newkp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # ax1 = fig.add_subplot(121), plt.imshow(img1)
    # ax2 = fig.add_subplot(122), plt.imshow(img2)
    # ax1= plt.subplot(1,2,1), plt.imshow(img1)
    # ax2= plt.subplot(1,2,2), plt.imshow(img2)
    # for i in range(len(newkp)):
    #     line = matplotlib.lines.Line2D((newkp[i].pt[0],newkp2[i].pt[0]),(newkp[i].pt[1],newkp2[i].pt[1]))
    #     fig.lines.append(line)

    # plt.show()

    # matches = []    
    # for i in cord:
    #     if (i[0]>0 and i[1]>0):
    #         matches.append(cv2.DMatch(_queryIdx = i[1], _trainIdx = i[0],  _imgIdx = 0, _distance = i[2]))
    #     else:
    #         print ("huh")
    # print(len(matches))
    

    #         # if (j != 0):
    #         #     dist2 = get_distance(kp[i].pt, kp2[j-1].pt)
    #         # if (j != (len(kp) - 1)):
    #         #     dist3 = get_distance(kp[i].pt, kp2[j+1].pt)
    #         # if (dist < dist2 ) and (dist < dist3):
    #         #     ratio = dist/min(dist2, dist3)
    #         #     if ratio < 0.8:
    #         #         cord.append((kp[i],kp2[j]))
    #         #     break
    


    # img3 = cv2.drawMatches(gray,newkp,gray2,newkp2,matches,outImg = None)
    # plt.imshow(img3),plt.show()


    # img=cv2.drawKeypoints(gray,kp,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img),plt.show()
    # img2=cv2.drawKeypoints(gray2,kp2,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img2),plt.show()
