import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  img = cv2.imread('./images/building.jpg')
  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


  surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 1000)
  kp = surf.detect(gray,None)


  img=cv2.drawKeypoints(gray,kp,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  plt.imshow(img),plt.show()