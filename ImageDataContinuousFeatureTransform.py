import cv2

class ImageDataContinuousFeatureTransform:

    def __init__(self, imagePath: str, scaleFactor):
        self.__path = imagePath
        self.__rawImageData = self.__getRawImageData(imagePath, scaleFactor)
        self.__foundKeyPoints, self.__foundDescriptors = self.__getFeatures()

    def __str__(self):
        return f"RawImageData={self.__rawImageData}; transformationMatrix={self.__transformationMatrix}; warpedPoints={self.__warpedPoints}"

    def __getFeatures(self):
        MIN_MATCH_COUNT = 10000
        sift = cv2.SIFT_create(MIN_MATCH_COUNT)
        return sift.detectAndCompute(cv2.cvtColor(self.__rawImageData, cv2.COLOR_BGR2GRAY),None)

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
    def path(self):
        return self.__path