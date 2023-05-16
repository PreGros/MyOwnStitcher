##########################################################################
##                                                                      ##
##          Autor: Tomáš Zaviačič                                       ##
##          Jméno souboru: ImageDataGpsTransform                        ##
##          Projekt: Automatická tvorba mapy dronem                     ##
##                                                                      ##
##########################################################################

import cv2
import numpy as np
import math
import time
from exif import Image as exifImage

# Nastavení globálních proměnných
GSD = -1
BASELONGITUDE = 0
BASELATITUDE = 0
BASEANGLE = -1

# Třída pro zpracování snímků pomocí metody polohovacích metadat 

class ImageDataGpsTransform:

    def __init__(self, imagePath: str, scaleFactor):
        self.__rawImageData = self.__getRawImageData(imagePath, scaleFactor)
        startComputing = time.time()
        self.__transformationMatrix = self.__getTransformationMatrix(imagePath)
        self.__timeSpent = time.time() - startComputing
        self.__warpedPoints = self.getWarpedPoints()

    def __str__(self):
        return f"RawImageData={self.__rawImageData}; transformationMatrix={self.__transformationMatrix}; warpedPoints={self.__warpedPoints}"

    # Uložení maticového zastoupení zpracovávaného snímku, který může být zmenšený podle vstupních argumentů
    def __getRawImageData(self, imgPath, scaleFactor):
        if (scaleFactor == 1.0):
            return cv2.imread(imgPath)
        else:
            rawImgData = cv2.imread(imgPath)
            width = int(rawImgData.shape[1] * scaleFactor)
            height = int(rawImgData.shape[0] * scaleFactor)
            return cv2.resize(rawImgData, (width, height))
    
    # Přepočítání formátu 0 až -180 stupňů na formát 0 až 360 stupňů
    def __convertAngleToThreeSixty(self, angleToConvert):
        temp = 180 - abs(angleToConvert)
        return 180 + temp
    
    # Získání natočení dronu z xml dat snímku
    def __fetchFlightYawDegree(self, img):
        fd = open(img, encoding = 'latin-1')
        data = fd.read()
        xmpInvestigatedStart = data.find('drone-dji:FlightYawDegree="')
        data = data[xmpInvestigatedStart+len('drone-dji:FlightYawDegree="'):]
        xmpInvestigatedStart = data.find('"')
        angle = float(data[:xmpInvestigatedStart])

        if (angle < 0):
            angle = self.__convertAngleToThreeSixty(angle)

        return angle
    
    # Výpočet potřebného úhlu, který natočí aktuálně zpracovávaný snímek do natočení referenčního snímku ve směru proti hodinových ručiček
    def __determineAngleToBase(self, img):
        imgAngle = self.__fetchFlightYawDegree(img)

        if (imgAngle > BASEANGLE):
            return imgAngle - BASEANGLE
        else:
            return imgAngle + (360 - BASEANGLE)
        
    # Harvesinův vzorec převzatý z https://www.movable-type.co.uk/scripts/latlong.html
    def __getDistance(self, pt1, pt2):
        R = 6371e3 # metry
        lat1 = math.radians(pt1[0])
        lat2 = math.radians(pt2[0])
        Δlat = math.radians(pt2[0]-pt1[0])
        Δlon = math.radians(pt2[1]-pt1[1])

        a = math.sin(Δlat/2) * math.sin(Δlat/2) + math.cos(lat1) * math.cos(lat2) * math.sin(Δlon/2) * math.sin(Δlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c # v metrech
    
    # Převedení zeměpisných souřadnic ze stupnů, minut a vteřin na decimální stupně
    def __convertToDegrees(self, dmsValues):
        return dmsValues[0] + dmsValues[1] / 60.0 + dmsValues[2] / 3600.0
    
    # Získání relativní výšky dronu z xml dat snímku
    def __fetchRelativeAltitude(self, img):
        fd = open(img, encoding = 'latin-1')
        data = fd.read()
        xmpInvestigatedStart = data.find('drone-dji:RelativeAltitude="')
        data = data[xmpInvestigatedStart+len('drone-dji:RelativeAltitude="'):]
        xmpInvestigatedStart = data.find('"')
        data = data[:xmpInvestigatedStart]
        return float(data)
    
    # Výpočet GSD podle referečního snímku
    def __determineGroundSamplingDistance(self, imgPath):
        # focalLength = 0.5 # cm
        # width = self.__rawImageData.shape[1] # pixels
        # height = self.__rawImageData.shape[0] # pixels
        # sensorWidth = 0.616 # cm
        # sensorHeight = 0.455 # cm

        focalLength = 0.7 # cm
        width = self.__rawImageData.shape[1] # pixels
        height = self.__rawImageData.shape[0] # pixels
        sensorWidth = 1.0 # cm
        sensorHeight = 0.75 # cm

        baseAltitude = self.__fetchRelativeAltitude(imgPath)

        GSDh = ((baseAltitude*100)*sensorHeight)/(focalLength*height)
        GSDw = ((baseAltitude*100)*sensorWidth)/(focalLength*width)
        return GSDh if (GSDh > GSDw) else GSDw # in cm/px
        
    # Výpočet posunu podle polohovacích metadat a GSD
    def __determineShiftToBaseYX(self, imgPath):
        global GSD
        global BASELONGITUDE
        global BASELATITUDE
    
        # Načti zeměpisné souřadnice
        with open(imgPath, 'rb') as source:
            data = exifImage(source)
        latitude = self.__convertToDegrees(data.gps_latitude)
        longitude = self.__convertToDegrees(data.gps_longitude)
        latitude = latitude if (data.gps_latitude_ref == 'N') else -latitude
        longitude = longitude if (data.gps_longitude_ref == 'E') else -longitude

        # Pokud se jedná o první snímek spočítej GSD
        if (GSD == -1):
            GSD = self.__determineGroundSamplingDistance(imgPath)
            BASELONGITUDE = longitude
            BASELATITUDE = latitude
            return (0, 0)

        # Nastavení potřebných bodů pro výpočet 'x' a 'y' souřadnice posunu
        A = (BASELATITUDE, BASELONGITUDE)
        B = (BASELATITUDE, longitude)
        D = (latitude, BASELONGITUDE)

        # Výpočet vzdáleností mezi body
        AB = self.__getDistance(A,B) # x
        AD = self.__getDistance(A,D) # y

        # Vypočítaná vzdálenost bude podle Haversinova vzorce vždy kladná
        # Je tedy nutná kontrola, jestli nemá být nějaká hodnota záporná
        a = AB if (BASELONGITUDE<longitude) else -AB
        b = AD if (BASELATITUDE<latitude) else -AD

        # Vypočítané hodnoty 'a','b' jsou nyní posuny v x-ové a y-ové ose podle souřadného systému sever, východ..
        # Protože se výsledná mapa orientuje podle referenčního (prvního) snímku, musejí se posuny přepočítat.
        # Posuny se můžou vzít jako souřadnice bodu, pokud počátkem je referenční snímek. Pro tento bod se najdou nové souřadnice rotací souřadného systému kolem středu o úhel natočení referenčního snímku.
        x = a*math.cos(math.radians(BASEANGLE)) - b*math.sin(math.radians(BASEANGLE))
        y = a*math.sin(math.radians(BASEANGLE)) + b*math.cos(math.radians(BASEANGLE))

        # Přepočet vzdálenosti na pixely pomocí GSD
        translateX = int((x*100)/GSD)
        translateY = int((y*100)/GSD)*(-1) # -1 => úprava y-ové souřadnice, protože v maticovém zastoupení snímku je y-ová naopak

        return (translateY, translateX)
    
    # Výpočet transformační matice podle vypočítaného posunu a rotační matice
    def __determineTransformationMatrix(self, shiftToBaseYX, angleToBase):
        height, width = self.rawImageData.shape[:2]
        centerY, centerX = (height//2,width//2)
        rotM = cv2.getRotationMatrix2D((centerX, centerY), -1*angleToBase, 1.0)
        rotM = np.vstack((rotM, np.array([0, 0, 1])))

        translationM = np.float64([[1, 0, shiftToBaseYX[1]],
                                   [0, 1, shiftToBaseYX[0]],
                                   [0, 0,                1]])
        
        transformationMatrix = translationM @ rotM

        return transformationMatrix
        


    # Získání transformační matice výpočtem posunu a úhlu, o který se natočení aktuálního snímku dostane do stejného natočení jako má referenční snímek
    def __getTransformationMatrix(self, imgPath):
        global BASEANGLE
        angleToBase = 0

        if (BASEANGLE == -1):
            BASEANGLE = self.__fetchFlightYawDegree(imgPath)
            # angleToBase = 0
        else:
            angleToBase = self.__determineAngleToBase(imgPath)

        shiftToBaseYX = self.__determineShiftToBaseYX(imgPath)

        transformationMatrix = self.__determineTransformationMatrix(shiftToBaseYX, angleToBase)

        return transformationMatrix


        
    # Předpočítání hran snímku deformované vypočítanou transformační maticí
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
    
    @property
    def timeSpent(self):
        return self.__timeSpent