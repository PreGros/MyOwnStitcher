from stitcher import stitchDataset
from stitcherFtS import stitchDatasetFt
from ImagesList import * 
import os.path
import argparse
import time

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
parser.add_argument('-ftS', help='create map using only feature matching', action='store_true')
parser.add_argument('-mask', help='should stitcher use mask to remove black corners on rotated images', action='store_true')
parser.add_argument('-outputName', metavar='-o', type=str, default="stitcherOutput", help='output file name')
parser.add_argument('-scale', type=float, default=1.0, help='scale factor to resize input images')

args = parser.parse_args()

# Start timer
start_time = time.time()

if ((args.mask == True and args.gpsS == False) and (args.mask == True and args.ftS == False)):
    parser.error('The -mask argument requires the -gpsS or -ftS argument')

if (args.gpsS == True and args.ftS == True):
    parser.error('The -gpsS argument and the -ftS cannot be simultaneously')

if (args.gpsS):
    imgDataList = ImagesList()
    imgDataList.runGPSTransform(args.infile, args.scale)
    stitchDataset(imgDataList.imageDataList, args.outputName, args.mask)

if (args.ftS):
    imgDataList = ImagesList()
    imgDataList.runFeatureTransform(args.infile, args.scale)
    stitchDatasetFt(imgDataList.imageDataList, args.outputName, args.mask)

# End timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time) 