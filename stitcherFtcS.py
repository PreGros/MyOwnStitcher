import numpy as np
import cv2

def getMinMax(resultMap, imgData, homography):
    resultMapHeight, resultMapWidth = resultMap.shape[:2]
    imageHeight, imageWidth = (imgData.rawImageData).shape[:2]

    resultMapCorners = np.float32([[0,0], [0, resultMapHeight], [resultMapWidth, resultMapHeight], [resultMapWidth, 0]]).reshape(-1,1,2)

    imageCorners = np.float32([[0,0], [0, imageHeight], [imageWidth, imageHeight], [imageWidth, 0]]).reshape(-1,1,2)
    imageWarpedCorners = cv2.perspectiveTransform(imageCorners, homography)

    concatPoints = np.concatenate((resultMapCorners, imageWarpedCorners), axis=0)  

    minXY = np.int32(concatPoints.min(axis=0).ravel() - 0.5)
    maxXY = np.int32(concatPoints.max(axis=0).ravel() + 0.5)

    return minXY, maxXY

def addMaskedImage(resultMap, backgroundImg):
    # Make a True/False mask of pixels whose BGR values sum to more than zero
    alpha = np.sum(resultMap, axis=-1) > 0

    # Convert True/False to 0/255 and change type to "uint8" to match "na"
    alpha = np.uint8(alpha * 255)

    # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
    res = np.dstack((resultMap, alpha))

    # extract alpha channel from foreground image as mask and make 3 channels
    alpha = res[:,:,3]
    alpha = cv2.merge([alpha,alpha,alpha])

    # extract bgr channels from foreground image
    front = res[:,:,0:3]


    return np.where(alpha==(0,0,0), backgroundImg, front)

def stitchDatasetFtc(imgDataList, outputName, maskFlag):

    if (len(imgDataList) < 1):
        raise Exception("Stitcher need at least one image")

    reprojectionThreshold = 5.0

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    MIN_MATCH_COUNT = 10000
    sift = cv2.SIFT_create(MIN_MATCH_COUNT)


    resultMap = imgDataList[0].rawImageData

    for i in range(len(imgDataList) - 1):
        imgData = imgDataList[i+1]

        print("Zpracovává se obrázek číslo {0} z {1}".format(i+2, len(imgDataList)))
        print(imgData.path)

        


        
        if (i == 0):
            foundKeyPoints = imgDataList[0].foundKeyPoints
            foundDescriptors = imgDataList[0].foundDescriptors
        else:
            try:
                foundKeyPoints, foundDescriptors = sift.detectAndCompute(cv2.cvtColor(resultMap, cv2.COLOR_BGR2GRAY),None)
            except:
                cv2.imwrite("outputMosaics/{0}.png".format(outputName), resultMap)
                print("FAIL")
                quit()


        print(len(foundKeyPoints))
        print(len(imgData.foundKeyPoints))

        matches = flann.knnMatch(foundDescriptors, imgData.foundDescriptors,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        dst_pts = np.float32([ foundKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ imgData.foundKeyPoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        try:
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,reprojectionThreshold)
        except:
            cv2.imwrite("outputMosaics/{0}.png".format(outputName), resultMap)
            print("Mezi snímky nebylo nalezeno dostatek obrazových příznaků!")
            quit()
             




        minXY, maxXY = getMinMax(resultMap, imgData, homography)

        translation_dist = [-minXY[0],-minXY[1]]
        H_translation    = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

        combinedMatrix =  H_translation.dot(homography)

        outputImg = cv2.warpPerspective(imgData.rawImageData, combinedMatrix, (maxXY[0]-minXY[0], maxXY[1]-minXY[1]))

        if (maskFlag and i != 0):
            backgroundImg = outputImg[translation_dist[1]:translation_dist[1]+resultMap.shape[0], translation_dist[0]:translation_dist[0]+resultMap.shape[1]]
            resultMap = addMaskedImage(resultMap, backgroundImg)

        outputImg[translation_dist[1]:translation_dist[1]+resultMap.shape[0], translation_dist[0]:translation_dist[0]+resultMap.shape[1]] = resultMap

        resultMap = outputImg

    cv2.imwrite("outputMosaics/{0}.png".format(outputName), resultMap)