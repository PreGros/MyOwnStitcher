from stitcher import stitchDataset
from stitcherFtcS import stitchDatasetFtc
from ImagesList import * 
import os.path
import argparse
import time


# kontrola vstupní složky
def fileValidity(parser, arg):
    if arg == "":
        return ""
    elif not os.path.exists(arg):
        parser.error("The file \'{0}\' does not exist!".format(arg))
    else:
        return arg


# Zpracovávání vstupních argumentů

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('infile', nargs='?',
                    default="", type=lambda x: fileValidity(parser, x), 
                    help='input folder containing unstitched photos')
parser.add_argument('-gpsS', help='create map using gps information', action='store_true')
parser.add_argument('-ftS', help='create map using feature matching', action='store_true')
parser.add_argument('-ftcS', help='create map using continuous map stitching', action='store_true')
parser.add_argument('-mask', help='should stitcher use mask to remove black corners on rotated images', action='store_true')
parser.add_argument('-outputName', metavar='-o', type=str, default="stitcherOutput", help='output file name')
parser.add_argument('-scale', type=float, default=1.0, help='scale factor to resize input images')

args = parser.parse_args()
start_time = time.time()

# Pokud se použil 'mask' argument bez metody

if ((args.mask == True and args.gpsS == False) and (args.mask == True and args.ftS == False) and (args.mask == True and args.ftcS == False)):
    parser.error('The -mask argument requires the -gpsS or -ftS argument')


# Pokud byly zadány více jak dvě metody

if ((int(args.gpsS) + int(args.ftS) + int(args.ftcS)) > 1):
    parser.error('Arguments -gpsS, -ftS and ftcS cannot be simultaneously')


# Spusť vytvoření pomocí polohovacích metadat

if (args.gpsS):
    imgDataList = ImagesList()
    imgDataList.runGPSTransform(args.infile, args.scale)
    stitchDataset(imgDataList.imageDataList, imgDataList.timeSpent, args.outputName, args.mask)

# Spusť vytvoření pomocí obrazových příznaků mezi snímky

if (args.ftS):
    imgDataList = ImagesList()
    imgDataList.runFeatureTransform(args.infile, args.scale)
    stitchDataset(imgDataList.imageDataList, imgDataList.timeSpent, args.outputName, args.mask)


# Spusť vytvoření pomocí postupného skládání mapy

if (args.ftcS):
    imgDataList = ImagesList()
    imgDataList.runFeatureContinuousTransform(args.infile, args.scale)
    stitchDatasetFtc(imgDataList.imageDataList, imgDataList.timeSpent, args.outputName, args.mask)


print("Celkový čas: ", time.time() - start_time) 