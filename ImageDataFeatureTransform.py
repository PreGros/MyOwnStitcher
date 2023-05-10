import cv2
import numpy as np

class ImageDataFeatureTransform:

    def __init__(self, imagePath: str, prevImageData: 'ImageDataFeatureTransform', scaleFactor):
        self.__path = imagePath
        self.__rawImageData = self.__getRawImageData(imagePath, scaleFactor)
        self.__foundKeyPoints, self.__foundDescriptors = self.__getFeatures()

    def __str__(self):
        return f"RawImageData={self.__rawImageData}; transformationMatrix={self.__transformationMatrix}; warpedPoints={self.__warpedPoints}"

    def __getFeatures(self):
        MIN_MATCH_COUNT = 10000
        sift = cv2.SIFT_create(MIN_MATCH_COUNT)
        return sift.detectAndCompute(cv2.cvtColor(self.__rawImageData, cv2.COLOR_BGR2GRAY),None)

    # def __getHomography(self, prevImageData: 'ImageDataFeatureTransform'):
    #     reprojectionThreshold = 5.0

    #     FLANN_INDEX_KDTREE = 0
    #     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #     search_params = dict(checks = 50)

    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
    #     matches = flann.knnMatch(prevImageData.__foundDescriptors,self.__foundDescriptors,k=2)

    #     # store all the good matches as per Lowe's ratio test.
    #     good = []
    #     for m,n in matches:
    #         if m.distance < 0.3*n.distance:
    #             good.append(m)

    #     dst_pts = np.float32([ prevImageData.__foundKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     src_pts = np.float32([ self.__foundKeyPoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #     try:
    #         H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,reprojectionThreshold)
    #     except:
    #         print("Mezi snímky nebylo nalezeno dostatek obrazových příznaků!")
    #         quit()

    #     return H

    def __getRawImageData(self, imgPath, scaleFactor):
        if (scaleFactor == 1.0):
            return cv2.imread(imgPath)
        else:
            rawImgData = cv2.imread(imgPath)
            width = int(rawImgData.shape[1] * scaleFactor)
            height = int(rawImgData.shape[0] * scaleFactor)
            return cv2.resize(rawImgData, (width, height))

    # def __getTransformationMatrix(self, prevImageData: 'ImageDataFeatureTransform'): 
    #     if (prevImageData != None):
    #         homographyMatrix = self.__getHomography(prevImageData)
    #         return prevImageData.__transformationMatrix @ homographyMatrix
    #     else:
    #         T0 = np.float64([
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1]])
    #         return T0

    # def __getWarpedPoints(self):
    #     height, width = (self.__rawImageData).shape[:2]
    #     points = np.float32([[0,0], [0, height], [width,height], [width, 0]]).reshape(-1,1,2)
    #     return cv2.perspectiveTransform(points, self.__transformationMatrix)
    
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