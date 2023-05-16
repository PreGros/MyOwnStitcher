##########################################################################
##                                                                      ##
##          Autor: Tomáš Zaviačič                                       ##
##          Jméno souboru: stitcher                                     ##
##          Projekt: Automatická tvorba mapy dronem                     ##
##                                                                      ##
##########################################################################

import numpy as np
import cv2

# Funkce pro zjištění minimálních a maximálních souřadnic dvou snímků po jejich deformaci homografií
def getMinMax(imgDataList):
    concatPoints = imgDataList[0].warpedPoints
    for i in range(len(imgDataList)-1):
        i = i + 1
        concatPoints = np.concatenate((concatPoints, imgDataList[i].warpedPoints), axis=0)

    minXY = np.int32(concatPoints.min(axis=0).ravel() - 0.5)
    maxXY = np.int32(concatPoints.max(axis=0).ravel() + 0.5)

    return minXY, maxXY

# Vytváření masky a vkládání snímku na pozadí pomocí vytvořené masky
def maskImage(img, ouputImg, backgroundImg, combinedMatrix):
    rows, cols = img.shape[:2]
    points_default = np.float32([[0,0], [0,rows], [cols,rows], [cols,0]]).reshape(-1,1,2)
    correctedPoints = cv2.perspectiveTransform(points_default, combinedMatrix)

    correctedPoints = np.around(correctedPoints)

    mask = np.zeros_like(ouputImg)
    mask = cv2.fillPoly(mask,[correctedPoints.astype(int)], (255,255,255))

    # Tato část vkládá snímek na pozadí přes masku
    # Je inspirovaná podle https://stackoverflow.com/questions/41572887/equivalent-of-copyto-in-python-opencv-bindings/41573727#41573727
    locs = np.where(mask != 0) # Get the non-zero mask locations 
    backgroundImg[locs[0], locs[1]] = ouputImg[locs[0], locs[1]]
    
    return backgroundImg

# Spojování snímků pomocí transformačních matic
def stitchDataset(imgDataList, timeSpent, outputName, maskFlag):

    if (len(imgDataList) < 1):
        raise Exception("Pro vytvoření mapy je potřeba alespoň jeden snímek.")

    # Vytvoření prázdné plochy o potřebné velikosti
    canvasMinXY, canvasMaxXY = getMinMax(imgDataList)
    canvas = np.zeros((canvasMaxXY[1]-canvasMinXY[1],canvasMaxXY[0]-canvasMinXY[0],3),dtype=int)

    # Hodnota, která definuje pozici referenčního snímku na výsledné mapě
    shiftToStartYX = (abs(canvasMinXY[1]), abs(canvasMinXY[0]))

    # Procházej od posledního snímku
    for i in range(len(imgDataList)):
        imgData = imgDataList.pop()
        
        img = imgData.rawImageData

        # Zjištění minima, maxima aktuálně zpracovávaného snímku
        imgMinXY   = np.int32(imgData.warpedPoints.min(axis=0).ravel() - 0.5)
        imgMaxXY   = np.int32(imgData.warpedPoints.max(axis=0).ravel() + 0.5)

        # Upravení transformace podle minima
        zaporPosun = [-imgMinXY[0],-imgMinXY[1]]
        maticeZaporPosun = np.array([[1, 0, zaporPosun[0]], [0, 1, zaporPosun[1]], [0,0,1]])
        combinedMatrix =  maticeZaporPosun.dot(imgData.transformationMatrix)

        # Transformace zpracovávaného snímku podle upravené homografie
        outputImg = cv2.warpPerspective(img, combinedMatrix, (imgMaxXY[0]-imgMinXY[0], imgMaxXY[1]-imgMinXY[1]))

        # Vymaskování
        if (maskFlag):
            backgroundImg = canvas[shiftToStartYX[0]+imgMinXY[1]:shiftToStartYX[0]+imgMinXY[1]+outputImg.shape[0], shiftToStartYX[1]+imgMinXY[0]:shiftToStartYX[1]+imgMinXY[0]+outputImg.shape[1]]
            outputImg = maskImage(img, outputImg, backgroundImg, combinedMatrix)

        # Vložení transformovaného snímku na mapu
        canvas[shiftToStartYX[0]+imgMinXY[1]:shiftToStartYX[0]+imgMinXY[1]+outputImg.shape[0], shiftToStartYX[1]+imgMinXY[0]:shiftToStartYX[1]+imgMinXY[0]+outputImg.shape[1]] = outputImg

    cv2.imwrite("outputMosaics/{0}.png".format(outputName), canvas)
    print("Výpočet transformací u všech snímků trval průměrně {0} sekund".format(timeSpent))