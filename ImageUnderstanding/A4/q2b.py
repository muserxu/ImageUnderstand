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