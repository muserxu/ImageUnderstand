#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load the Drive helper and mount
# from google.colab import drive

# # This will prompt for authorization.
# drive.mount('/content/drive')


# # In[ ]:


# # install OpenCV
# get_ipython().system(u'pip install opencv-python')

# # the usual ...
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import cv2
import numpy as np


# In[ ]:


# read an image (make sure CavePainting.jpg is in the main folder in your Google Drive)
# img = cv2.imread('/images/building.jpg') # READS IN NUMPY ARRAY


# In[ ]:


# let's make a function
def imshowBGR2RGB( im ):
  img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.axis('off')
  return


# In[ ]:

if __name__ == '__main__':
  img = cv2.imread('./images/building.jpg')

  imshowBGR2RGB(img)


  # In[ ]:


  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),7)
  Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
  Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)


  # In[ ]:


  plt.subplot(2,1,1), plt.imshow(Ix,cmap = 'gray')
  plt.subplot(2,1,2), plt.imshow(Iy,cmap = 'gray')


  # In[ ]:


  IxIy = np.multiply(Ix, Iy)
  Ix2 = np.multiply(Ix, Ix)
  Iy2 = np.multiply(Iy, Iy)


  # In[ ]:


  Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
  Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
  IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 


  # In[ ]:


  plt.subplot(1,3,1), plt.imshow(Ix2_blur,cmap = 'gray')
  plt.subplot(1,3,2), plt.imshow(Iy2_blur,cmap = 'gray')
  plt.subplot(1,3,3), plt.imshow(IxIy_blur,cmap = 'gray')


  # In[ ]:


  det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
  trace = Ix2_blur + Iy2_blur


  # In[ ]:


  plt.subplot(1,2,1), plt.imshow(det,cmap = 'gray')
  plt.subplot(1,2,2), plt.imshow(trace,cmap = 'gray')
  


  # In[ ]:


  R = det - 0.05 * np.multiply(trace,trace)
  plt.subplot(1,2,1), plt.imshow(img), plt.axis('off')
  plt.subplot(1,2,2), plt.imshow(R,cmap = 'gray'), plt.axis('off')
  plt.show()

