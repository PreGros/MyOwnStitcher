from stitcher import stitchDataset
from ImagesList import * 
import os.path
import argparse

def fileValidity(parser, arg):
    if arg == "":
        return ""
    elif not os.path.exists(arg):
        parser.error("The file \'{0}\' does not exist!".format(arg))
    else:
        return arg





parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('infile', nargs='?',
                    default="", type=lambda x: fileValidity(parser, x), 
                    help='input folder containing unstitched photos')
parser.add_argument('-gpsS', help='create map using only gps information', action='store_true')
parser.add_argument('-ftS', help='create map using only gps information, rotated', action='store_true')
parser.add_argument('-mask', help='should gps stitcher use mask to remove black corners on rotated images', action='store_true')
parser.add_argument('-outputName', metavar='-o', type=str, default="stitcherOutput")

args = parser.parse_args()

if ((args.mask == True and args.gpsS == False) and (args.mask == True and args.ftS == False)):
    parser.error('The -mask argument requires the -gpsS argument')

if (args.gpsS == True and args.ftS == True):
    parser.error('The -gpsS argument and the -ftS cannot be simultaneously')

if (args.gpsS):
    imgDataList = ImagesList()
    imgDataList.runGPSTransform(args.infile)
    stitchDataset(imgDataList.imageDataList, args.outputName, args.mask)

if (args.ftS):
    imgDataList = ImagesList()
    imgDataList.runFeatureTransform(args.infile)
    stitchDataset(imgDataList.imageDataList, args.outputName, args.mask)
