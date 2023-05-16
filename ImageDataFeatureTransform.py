import cv2
import numpy as np
import time

# Třída pro zpracování snímků pomocí metody obrazových příznaků mezi snímky

class ImageDataFeatureTransform:

    def __init__(self, imagePath: str, prevImageData: 'ImageDataFeatureTransform', scaleFactor):
        self.__rawImageData = self.__getRawImageData(imagePath, scaleFactor)
        startCompute = time.time()
        self.__foundKeyPoints, self.__foundDescriptors = self.__getFeatures()
        self.__transformationMatrix = self.__getTransformationMatrix(prevImageData, imagePath)
        self.__timeSpent = time.time() - startCompute
        self.__warpedPoints = self.__getWarpedPoints()

    def __str__(self):
        return f"RawImageData={self.__rawImageData}; transformationMatrix={self.__transformationMatrix}; warpedPoints={self.__warpedPoints}"

    # Detekování obrazových příznaků pomoc SIFT algoritmu
    def __getFeatures(self):
        MIN_MATCH_COUNT = 10000
        sift = cv2.SIFT_create(MIN_MATCH_COUNT)
        return sift.detectAndCompute(cv2.cvtColor(self.__rawImageData, cv2.COLOR_BGR2GRAY),None)

    # Výpočet homografie
    def __getHomography(self, prevImageData: 'ImageDataFeatureTransform', imagePath: str):

        # Nastaví práhu pro reprojekční error
        reprojectionThreshold = 5.0

        # Inicializace FLANN
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        # Vypočítej nejpodobnější dvojice pomocí knihovny FLANN
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(prevImageData.__foundDescriptors,self.__foundDescriptors,k=2)

        # Nad dvojicemi proveď Lowe's ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.3*n.distance:
                good.append(m)

        # Dst body jsou zájmové body předešlého snímku patřící do dobrých dvojic (good)
        # Src body jsou zájmové body zpracovávaného snímku patřící do dobrých dvojic (good)
        dst_pts = np.float32([ prevImageData.__foundKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ self.__foundKeyPoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        try:
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,reprojectionThreshold)
        except:
            print("U snímku s názvem \'{0}\' se nepodařilo najít obrazové příznaky!".format(imagePath))
            quit()

        return H

    # Uložení maticového zastoupení zpracovávaného snímku, který může být zmenšený podle vstupních argumentů
    def __getRawImageData(self, imgPath, scaleFactor):
        if (scaleFactor == 1.0):
            return cv2.imread(imgPath)
        else:
            rawImgData = cv2.imread(imgPath)
            width = int(rawImgData.shape[1] * scaleFactor)
            height = int(rawImgData.shape[0] * scaleFactor)
            return cv2.resize(rawImgData, (width, height))

    # Výpočet transformační matice vynásobením transformační matice předchozí snímku s vypočítanou homografií
    def __getTransformationMatrix(self, prevImageData: 'ImageDataFeatureTransform', imagePath: str): 
        if (prevImageData != None):
            homographyMatrix = self.__getHomography(prevImageData, imagePath)
            return prevImageData.__transformationMatrix @ homographyMatrix
        else:
            T0 = np.float64([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
            return T0

    # Předpočítání hran snímku deformované vypočítanou transformační maticí
    def __getWarpedPoints(self):
        height, width = (self.__rawImageData).shape[:2]
        points = np.float32([[0,0], [0, height], [width,height], [width, 0]]).reshape(-1,1,2)
        return cv2.perspectiveTransform(points, self.__transformationMatrix)
    
    @property
    def rawImageData(self):
        return self.__rawImageData
    
    @property
    def transformationMatrix(self):
        return self.__transformationMatrix
    
    @property
    def warpedPoints(self):
        return self.__warpedPoints
    
    @property
    def timeSpent(self):
        return self.__timeSpent