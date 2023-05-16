import cv2
import time

# Třída pro zpracování snímků pomocí metody postupného skládání mapy 

class ImageDataContinuousFeatureTransform:

    def __init__(self, imagePath: str, scaleFactor):
        startDetect = time.time()
        self.__rawImageData = self.__getRawImageData(imagePath, scaleFactor)
        self.__timeSpent = time.time() - startDetect
        self.__foundKeyPoints, self.__foundDescriptors = self.__getFeatures()

    def __str__(self):
        return f"RawImageData={self.__rawImageData}; transformationMatrix={self.__transformationMatrix}; warpedPoints={self.__warpedPoints}"

    # Detekování obrazových příznaků pomoc SIFT algoritmu
    def __getFeatures(self):
        MIN_MATCH_COUNT = 10000
        sift = cv2.SIFT_create(MIN_MATCH_COUNT)
        return sift.detectAndCompute(cv2.cvtColor(self.__rawImageData, cv2.COLOR_BGR2GRAY),None)

    # Uložení maticového zastoupení zpracovávaného snímku, který může být zmenšený podle vstupních argumentů
    def __getRawImageData(self, imgPath, scaleFactor):
        if (scaleFactor == 1.0):
            return cv2.imread(imgPath)
        else:
            rawImgData = cv2.imread(imgPath)
            width = int(rawImgData.shape[1] * scaleFactor)
            height = int(rawImgData.shape[0] * scaleFactor)
            return cv2.resize(rawImgData, (width, height))
    
    @property   
    def rawImageData(self):
        return self.__rawImageData
    
    @property
    def foundKeyPoints(self):
        return self.__foundKeyPoints
    
    @property
    def foundDescriptors(self):
        return self.__foundDescriptors
    
    @property
    def timeSpent(self):
        return self.__timeSpent