from ImageDataContinuousFeatureTransform import *
from ImageDataFeatureTransform import *
from ImageDataGpsTransform import *
import glob

class ImagesList:

    def __init__(self):
        self.imageDataList = []
        self.timeSpent = 0

    def runFeatureTransform(self, path: str, scaleFactor):
        pathsList = sorted(glob.glob("{0}/*.jpg".format(path)))

        for imgPath in pathsList:
            self.imageDataList.append(ImageDataFeatureTransform(imgPath,
                                                                None if (len(self.imageDataList) == 0) else self.imageDataList[-1], # předchozí prvek
                                                                scaleFactor) 
                                     )
            self.timeSpent = self.timeSpent + self.imageDataList[-1].timeSpent
        
        self.timeSpent = self.timeSpent / len(self.imageDataList)

    def runGPSTransform(self, path: str, scaleFactor):
        pathsList = sorted(glob.glob("{0}/*.jpg".format(path)))

        for imgPath in pathsList:
            self.imageDataList.append(ImageDataGpsTransform(imgPath, scaleFactor))
            self.timeSpent = self.timeSpent + self.imageDataList[-1].timeSpent
        
        self.timeSpent = self.timeSpent / len(self.imageDataList)
        
    def runFeatureContinuousTransform(self, path: str, scaleFactor):
        pathsList = sorted(glob.glob("{0}/*.jpg".format(path)))

        for imgPath in pathsList:
            self.imageDataList.append(ImageDataContinuousFeatureTransform(imgPath, scaleFactor))
            self.timeSpent = self.timeSpent + self.imageDataList[-1].timeSpent

        self.timeSpent = self.timeSpent / len(self.imageDataList)
            
        