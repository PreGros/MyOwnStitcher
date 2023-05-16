##########################################################################
##                                                                      ##
##          Autor: Tomáš Zaviačič                                       ##
##          Jméno souboru: stitcherFtcS                                 ##
##          Projekt: Automatická tvorba mapy dronem                     ##
##                                                                      ##
##########################################################################

import numpy as np
import cv2
import time

# Globální proměnné pro kontrolu rozměrů aktuální mapy

PREVMAPYX = 0
IMGBIGGERSIZE = 0


# Funkce pro zjištění minimálních, maximálních souřadnic mapy a snímku po jeho deformaci homografií
def getMinMax(resultMap, imgData, homography):
    resultMapHeight, resultMapWidth = resultMap.shape[:2]
    imageHeight, imageWidth = (imgData.rawImageData).shape[:2]

    resultMapCorners = np.float32([[0,0], [0, resultMapHeight], [resultMapWidth, resultMapHeight], [resultMapWidth, 0]]).reshape(-1,1,2)

    imageCorners = np.float32([[0,0], [0, imageHeight], [imageWidth, imageHeight], [imageWidth, 0]]).reshape(-1,1,2)
    imageWarpedCorners = cv2.perspectiveTransform(imageCorners, homography)

    concatPoints = np.concatenate((resultMapCorners, imageWarpedCorners), axis=0)  

    minXY = np.int32(concatPoints.min(axis=0).ravel() - 0.5)
    maxXY = np.int32(concatPoints.max(axis=0).ravel() + 0.5)

    return minXY, maxXY


# Vložení vymaskované části zpracovávaného na aktuální mapu 
# Je inspirováno podle https://stackoverflow.com/questions/69620706/overlay-image-on-another-image-with-opencv-and-numpy a https://stackoverflow.com/questions/70223829/opencv-how-to-convert-all-black-pixels-to-transparent-and-save-it-to-png-file
def addMaskedImage(resultMap, backgroundImg):
    # Vytvoř masku s True/False
    alpha = np.sum(resultMap, axis=-1) > 0
    # Převeď True/False na hodnoty (0,255)
    alpha = np.uint8(alpha * 255)
    # Vlož vrstvu alpha na vrstvy snímku pro převod z BGR na BGRA (ze 3 na 4)
    res = np.dstack((resultMap, alpha))

    alpha = res[:,:,3]
    alpha = cv2.merge([alpha,alpha,alpha])
    front = res[:,:,0:3]

    return np.where(alpha==(0,0,0), backgroundImg, front)


# Kontrola výsledných rozměrů aktuální mapy
# Největší rozměr aktuální mapy nesmí být větší než největší rozměr jednoho snímku plus polovina největšího rozměru jednoho snímku vynásobená počtem přidaných snímků na mapu
def checkResultMapDim(resultMapYX, currImgYX, i):
    global IMGBIGGERSIZE

    if (IMGBIGGERSIZE == 0):
        IMGBIGGERSIZE = currImgYX[0] if (currImgYX[0] > currImgYX[1]) else currImgYX[1]

    mapBiggerSize = resultMapYX[0] if (resultMapYX[0] > resultMapYX[1]) else resultMapYX[1]
    limitDim = IMGBIGGERSIZE+(IMGBIGGERSIZE*0.5*(i+1))

    return mapBiggerSize > limitDim


# Funkce pro postupnou tvorbu mapy
def stitchDatasetFtc(imgDataList, timeDetect, outputName, maskFlag):

    if (len(imgDataList) < 1):
        raise Exception("Je potřeba alespoň jeden snímek!")
    
    global PREVMAPYX

    # Nastaví práhu pro reprojekční error
    reprojectionThreshold = 5.0

    # Inicializace FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Inicializace SIFT
    MIN_MATCH_COUNT = 10000
    sift = cv2.SIFT_create(MIN_MATCH_COUNT)

    i = 0
    sumTime = 0

    # Nastavení prvního snímku jako výsledná mapa
    resultMap = imgDataList[0].rawImageData


    # Procházej od druhého snímku po poslední
    for i in range(len(imgDataList) - 1):
        imgData = imgDataList[i+1]        

        startDetect = time.time()
        
        # Detekuj obrazové příznaky
        if (i == 0):
            foundKeyPoints = imgDataList[0].foundKeyPoints
            foundDescriptors = imgDataList[0].foundDescriptors
        else:
            try:
                foundKeyPoints, foundDescriptors = sift.detectAndCompute(cv2.cvtColor(resultMap, cv2.COLOR_BGR2GRAY),None)
            except:
                cv2.imwrite("outputMosaics/{0}.png".format(outputName), resultMap)
                print("FAIL")
                quit()

        # Vypočítej nejpodobnější dvojice
        matches = flann.knnMatch(foundDescriptors, imgData.foundDescriptors,k=2)

        # Nad dvojicemi proveď Lowe's ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        # Dst body jsou zájmové body aktuální mapy patřící do dobrých dvojic (good)
        # Src body jsou zájmové body zpracovávaného snímku patřící do dobrých dvojic (good)
        dst_pts = np.float32([ foundKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ imgData.foundKeyPoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        try:
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,reprojectionThreshold)
        except:
            cv2.imwrite("outputMosaics/{0}.png".format(outputName), resultMap)
            print("Mezi snímky nebylo nalezeno dostatek obrazových příznaků!")
            quit()
             
        sumTime = sumTime + (time.time() - startDetect)


        # Získej minima a maxima aktuální mapy a zpracovávaného snímku po deformaci
        minXY, maxXY = getMinMax(resultMap, imgData, homography)

        # Upravení transformace podle minima
        zaporPosun = [-minXY[0],-minXY[1]]
        maticeZaporPosun    = np.array([[1, 0, zaporPosun[0]], [0, 1, zaporPosun[1]], [0,0,1]])
        combinedMatrix =  maticeZaporPosun.dot(homography)

        # Transformace zpracovávaného snímku a nastavení minimálního rozměru podle minima
        outputImg = cv2.warpPerspective(imgData.rawImageData, combinedMatrix, (maxXY[0]-minXY[0], maxXY[1]-minXY[1]))

        # Vložení vymaskované části zpracovávaného na aktuální mapu 
        if (maskFlag and i != 0):
            backgroundImg = outputImg[zaporPosun[1]:zaporPosun[1]+resultMap.shape[0], zaporPosun[0]:zaporPosun[0]+resultMap.shape[1]]
            resultMap = addMaskedImage(resultMap, backgroundImg)

        # Vložení aktuální mapy do transformovaného snímku
        outputImg[zaporPosun[1]:zaporPosun[1]+resultMap.shape[0], zaporPosun[0]:zaporPosun[0]+resultMap.shape[1]] = resultMap

        # Nastavení aktuální mapy
        resultMap = outputImg   

        # Kontrola výsledných rozměrů aktuální mapy
        if (checkResultMapDim(resultMap.shape[:2], (imgData.rawImageData).shape[:2], i)):
            print("CHYBA! - Velikost výsledné mapy se zvětšila z {0}x{1} na {2}x{3} pixelů! Chybná mapa byla uložena jako \'{4}_dimError\'".format(PREVMAPYX[1], 
                                                                                                                                               PREVMAPYX[0],
                                                                                                                                               resultMap.shape[1],
                                                                                                                                               resultMap.shape[0],
                                                                                                                                               outputName))
            print("Výpočet transformací u všech snímků trval průměrně {0} sekund".format((sumTime / (i+1)) + timeDetect))
            cv2.imwrite("outputMosaics/{0}_errorDim.png".format(outputName), resultMap)
            quit()

        # Nastavení globální proměnné pro případný výpis
        PREVMAPYX = resultMap.shape[:2]

    cv2.imwrite("outputMosaics/{0}.png".format(outputName), resultMap)
    print("Výpočet transformací u všech snímků trval průměrně {0} sekund".format((sumTime / (i+1)) + timeDetect))