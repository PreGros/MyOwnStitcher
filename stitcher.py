import numpy as np
import cv2

def getMinMax(imgDataList):
    concatPoints = imgDataList[0].warpedPoints
    for i in range(len(imgDataList)-1):
        i = i + 1
        concatPoints = np.concatenate((concatPoints, imgDataList[i].warpedPoints), axis=0)

    minXY = np.int32(concatPoints.min(axis=0).ravel() - 0.5)
    maxXY = np.int32(concatPoints.max(axis=0).ravel() + 0.5)

    return minXY, maxXY


def maskImage(img, ouputImg, backgroundImg, combinedMatrix):
    # Vytváření masky
    rows, cols = img.shape[:2]
    points_default = np.float32([[0,0], [0,rows], [cols,rows], [cols,0]]).reshape(-1,1,2)
    correctedPoints = cv2.perspectiveTransform(points_default, combinedMatrix)

    [corr_x_min, corr_y_min]   = np.int32(correctedPoints.min(axis=0).ravel() - 0.5)
    [corr_x_max, corr_y_max]   = np.int32(correctedPoints.max(axis=0).ravel() + 0.5)

    correctedPoints = np.around(correctedPoints)

    mask = np.zeros_like(ouputImg)
    mask = cv2.fillPoly(mask,[correctedPoints.astype(int)], (255,255,255))

    locs = np.where(mask != 0) # Get the non-zero mask locations

    # Case #1 - Other image is grayscale and source image is colour
    if len(backgroundImg.shape) == 3 and len(ouputImg.shape) != 3:
        backgroundImg[locs[0], locs[1]] = ouputImg[locs[0], locs[1], None]
    # Case #2 - Both images are colour or grayscale
    elif (len(backgroundImg.shape) == 3 and len(ouputImg.shape) == 3) or \
    (len(backgroundImg.shape) == 1 and len(ouputImg.shape) == 1):
        backgroundImg[locs[0], locs[1]] = ouputImg[locs[0], locs[1]]
    # Otherwise, we can't do this
    else:
        raise Exception("Incompatible input and output dimensions")
    
    return backgroundImg


def stitchDataset(imgDataList, timeSpent, outputName, maskFlag):

    if (len(imgDataList) < 1):
        raise Exception("Pro vytvoření mapy je potřeba alespoň jeden snímek.")

    # First need to create canvas with enough size for all transformed images
    canvasMinXY, canvasMaxXY = getMinMax(imgDataList)
    canvas = np.zeros((canvasMaxXY[1]-canvasMinXY[1],canvasMaxXY[0]-canvasMinXY[0],3),dtype=int)

    shiftToStartYX = (abs(canvasMinXY[1]), abs(canvasMinXY[0]))

    for i in range(len(imgDataList)):
        imgData = imgDataList.pop()
        
        img = imgData.rawImageData

        imgMinXY   = np.int32(imgData.warpedPoints.min(axis=0).ravel() - 0.5)
        imgMaxXY   = np.int32(imgData.warpedPoints.max(axis=0).ravel() + 0.5)

        translation_dist = [-imgMinXY[0],-imgMinXY[1]]
        H_translation    = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

        combinedMatrix =  H_translation.dot(imgData.transformationMatrix)

        outputImg = cv2.warpPerspective(img, combinedMatrix, (imgMaxXY[0]-imgMinXY[0], imgMaxXY[1]-imgMinXY[1]))

        if (maskFlag):
            backgroundImg = canvas[shiftToStartYX[0]+imgMinXY[1]:shiftToStartYX[0]+imgMinXY[1]+outputImg.shape[0], shiftToStartYX[1]+imgMinXY[0]:shiftToStartYX[1]+imgMinXY[0]+outputImg.shape[1]]
            outputImg = maskImage(img, outputImg, backgroundImg, combinedMatrix)

        canvas[shiftToStartYX[0]+imgMinXY[1]:shiftToStartYX[0]+imgMinXY[1]+outputImg.shape[0], shiftToStartYX[1]+imgMinXY[0]:shiftToStartYX[1]+imgMinXY[0]+outputImg.shape[1]] = outputImg

    cv2.imwrite("outputMosaics/{0}.png".format(outputName), canvas)
    print("Výpočet transformací u všech snímků trval průměrně {0} sekund".format(timeSpent))