import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image

im = np.array(Image.open('004945.jpg'), dtype=np.uint8)
# im = np.array(Image.open('004964.jpg'), dtype=np.uint8)

# im = np.array(Image.open('005002.jpg'), dtype=np.uint8)
ax.imshow(im)
string = 'im1object.txt'
file = open(string, 'r')
for line in file:
    data = line.split(':')
    digit = [int(s) for s in data[1].split() if s.isdigit()]
    if data[0] == 'traffic light':
        rect = patches.Rectangle((digit[0],digit[1]),(digit[2] - digit[0]),(digit[3]-digit[1]),linewidth=1,edgecolor='b',facecolor='none')
        ax.text(digit[0],digit[1], 'traffic light', color = 'red', fontsize=12)
    elif data[0] == 'car':
        rect = patches.Rectangle((digit[0],digit[1]),(digit[2] - digit[0]),(digit[3]-digit[1]),linewidth=1,edgecolor='r',facecolor='none')
        ax.text(digit[0],digit[1], 'car', color = 'red', fontsize=12)
    elif data[0] == 'person':
        rect = patches.Rectangle((digit[0],digit[1]),(digit[2] - digit[0]),(digit[3]-digit[1]),linewidth=1,edgecolor='g',facecolor='none')
        ax.text(digit[0],digit[1], 'person', color = 'red', fontsize=12)
    else:
        rect = patches.Rectangle((digit[0],digit[1]),(digit[2] - digit[0]),(digit[3]-digit[1]),linewidth=1,edgecolor='cyan',facecolor='none')
        ax.text(digit[0],digit[1], 'bicycle', color = 'red', fontsize=12)

    ax.add_patch(rect)
plt.show()

