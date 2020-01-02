# from imageai.Detection import ObjectDetection
# import os

# execution_path = os.getcwd()

# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

from imageai.Detection import ObjectDetection
import os
import numpy as np

execution_path = os.getcwd()


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


custom_objects = detector.CustomObjects(person=True, traffic_light=True, bicycle=True, car=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "004945.jpg"), output_image_path=os.path.join(execution_path , "left1.jpg"), minimum_percentage_probability=30, display_percentage_probability=False, display_object_name=False)

file = open('im1object.txt', 'w')
for eachObject in detections:
    string = eachObject["name"] + ': ' 
    # string = eachObject["name"] + " : " + eachObject["percentage_probability"]+ " : " + eachObject["box_points"] 
    file.write(string)
    for number in eachObject["box_points"]:
        file.write(str(number))
        file.write(' ')
    file.write("\n")
file.close()

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# import numpy as np

# im = np.array(Image.open('image.jpg'), dtype=np.uint8)

# # Create figure and axes
# fig,ax = plt.subplots(1)

# # Display the image
# ax.imshow(im)

# # Create a Rectangle patch
# rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')

# # Add the patch to the Axes
# ax.add_patch(rect)

# rect = patches.Rectangle((100,200),40,30,linewidth=1,edgecolor='b',facecolor='none')

# # Add the patch to the Axes
# ax.add_patch(rect)

# plt.show()