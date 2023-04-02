from ImageDataFeatureTransform import *
from ImageDataGpsTransform import *
import glob

class ImagesList:

    def __init__(self):
        self.imageDataList = []

    def runFeatureTransform(self, path: str):
        pathsList = sorted(glob.glob("{0}/*.jpg".format(path)))

        for imgPath in pathsList:
            self.imageDataList.append(ImageDataFeatureTransform(imgPath,
                                                                None if (len(self.imageDataList) == 0) else self.imageDataList[-1]) # předchozí prvek
                                     )

    def runGPSTransform(self, path: str):
        pathsList = sorted(glob.glob("{0}/*.jpg".format(path)))

        for imgPath in pathsList:
            self.imageDataList.append(ImageDataGpsTransform(imgPath))
        
        
        