import cv2
import numpy as np

class ImageDataFeatureTransform:

    def __init__(self, imagePath: str, prevImageData: 'ImageDataFeatureTransform'):
        self.__rawImageData = self.getRawImageData(imagePath)
        self.__transformationMatrix = self.getTransformationMatrix(prevImageData)
        self.__warpedPoints = self.getWarpedPoints()

    def __str__(self):
        return f"RawImageData={self.__rawImageData}; transformationMatrix={self.__transformationMatrix}; warpedPoints={self.__warpedPoints}"

    def __getHomography(self, img1):
        MIN_MATCH_COUNT = 10000
        sift = cv2.SIFT_create(MIN_MATCH_COUNT)
        reprojectionThreshold = 5.0

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(self.__rawImageData, cv2.COLOR_BGR2GRAY),None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.3*n.distance:
                good.append(m)

        dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,reprojectionThreshold)

        return H

    def getRawImageData(self, imgPath):
        return cv2.imread(imgPath)

    def getTransformationMatrix(self, prevImageData: 'ImageDataFeatureTransform'): 
        if (prevImageData != None):
            homographyMatrix = self.__getHomography(prevImageData.__rawImageData)
            return prevImageData.__transformationMatrix @ homographyMatrix
        else:
            T0 = np.float64([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
            return T0

    def getWarpedPoints(self):
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