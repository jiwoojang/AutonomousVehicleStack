import os
import matplotlib.image as mpimg
import numpy as np
import cv2
from laneDetectionPipeline import renderLaneLines

def main():
    """
    Here we actually begin the image processing for all the test still images
    """
    images = os.listdir("test_images/")
    outPath =  "test_images_output/"

    for imageFile in images:
        # Read an image
        image = mpimg.imread("test_images/"+imageFile)
        
        # Detect lanes
        laneDetected = renderLaneLines(image)
        
        # Write an image
        cv2.imwrite(outPath+imageFile, cv2.cvtColor(laneDetected, cv2.COLOR_RGB2BGR))
        
if __name__ == '__main__':
    main()