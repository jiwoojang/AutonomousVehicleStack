import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from helpers import grayscale, gaussian_blur, canny, region_of_interest, weighted_img

"""
The actual lane detection pipeline
"""
def sortLine(line, leftLines, rightLines):
    slope = (line[3]-line[1]) /(line[2] - line[0])
    
    if slope < 0:
        leftLines.append(line)
    elif slope > 0:
        rightLines.append(line)
    
def buildLines(image):
    # Apply a gray scale
    grayScaleImage = grayscale(image)
    
    # Apply the gausian blur
    blurredImage = gaussian_blur(grayScaleImage, 15)

    # Apply Canny Edge detection
    cannyImage = canny(blurredImage, 70, 100)
    
    # Get lines from Hough space
    rho = 2
    theta = np.pi/180
    threshold = 1
    minLineLength = 15
    maxLineGap = 5
    
    lines = cv2.HoughLinesP(cannyImage, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)
    
    # Transform into numpy arrays for easy processing
    # Each array represents [x1, y1, x2, y2]
    nplines = []
    
    for l in lines:
        nplines.append(np.array([np.float32(l[0][0]), np.float32(l[0][1]), np.float32(l[0][2]), np.float32(l[0][3])]))
        
    nplines = [ l for l in nplines if 0.5 <= np.abs((l[3]-l[1])/(l[2] - l[0])) <= 2]
    
    # Sort the lines based on whether they are likely to be the left or right lane marker
    leftLaneLines = []
    rightLaneLines = []
    
    for l in nplines:
        sortLine(l, leftLaneLines, rightLaneLines)
    
    # Calculate the left lane line
    # We use the median here because it gives us better performance for video 
    leftLaneOffset = np.median([(l[1] - (l[3]-l[1])*l[0] /(l[2] - l[0])) for l in leftLaneLines]).astype(int)
    leftLaneSlope = np.median([((l[3]-l[1])/(l[2] - l[0])) for l in leftLaneLines])
    
    # We use basic line algebra here, y_n = slope * x_n + offset
    leftLaneLine = np.array([0, 
                             leftLaneOffset, 
                             -int(np.round(leftLaneOffset / leftLaneSlope)), 
                             0])
    
    # Calculate the right lane line
    rightLaneOffset = np.median([(l[1] - (l[3]-l[1])*l[0] /(l[2] - l[0])) for l in rightLaneLines]).astype(int)
    rightLaneSlope = np.median([((l[3]-l[1])/(l[2] - l[0])) for l in rightLaneLines])

    # Must account for image origin being at the top left corner here
    rightLaneLine = np.array([0, 
                              rightLaneOffset, 
                              int(np.round((cannyImage.shape[0] - rightLaneOffset) / rightLaneSlope)), 
                              cannyImage.shape[0]])
    
    return leftLaneLine, rightLaneLine

def renderLaneLines(image):
    # Build the geometry of the detected lines
    laneLines = buildLines(image)
    h = image.shape[0]
    w = image.shape[1]
    
    # Make a mask on which to draw just the lines
    mask = np.zeros(shape=(h, w))
    
    cv2.line(mask, (laneLines[0][0], laneLines[0][1]), (laneLines[0][2], laneLines[0][3]), [255,0,0], 4)
    cv2.line(mask, (laneLines[1][0], laneLines[1][1]), (laneLines[1][2], laneLines[1][3]), [255,0,0], 4)
    
    # Cull the mask with region of interest
    # We want the bottom 40% ish of the image
    p1 = (10, h-10)
    p2 = (10, int(h*0.6))
    p3 = (w-10, int(h*0.6))
    p4 = (w-10, h-10)
    
    cullVerts = np.array([[p1, p2, p3, p4]], dtype=np.int32)
    mask = region_of_interest(mask, cullVerts)
    
    rgbMask = np.uint8(mask)
    
    # Make the mask 3 channel, but the contents red
    if len(rgbMask.shape) is 2:
        rgbMask = np.dstack((rgbMask, np.zeros_like(rgbMask), np.zeros_like(rgbMask)))

    return weighted_img(rgbMask, image, 0.6, 1.0, 0)