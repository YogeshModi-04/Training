import cv2
import numpy as np

# Utility Functions
def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        hor = [np.zeros((height, width, 3), np.uint8)] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(rows):
            for c in range(cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

def initializeTrackbars(initialTracbarVals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", initialTracbarVals[0], 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", initialTracbarVals[1], 255, nothing)
    cv2.createTrackbar("Kernel Size", "Trackbars", initialTracbarVals[2], 30, nothing)

def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    KernelSize = cv2.getTrackbarPos("Kernel Size", "Trackbars")
    KernelSize = max(KernelSize, 1)  # Prevent zero size kernel
    return Threshold1, Threshold2, KernelSize

def nothing(x):
    pass

def findLargestSquare(contours, minArea=500, aspectRatioRange=(0.7, 1.3)):
    largest_area = 0
    largest_square = None

    for cnt in contours:
        # Use convex hull to account for potential perspective distortions
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        peri = cv2.arcLength(hull, True)
        
        # More flexible approximation of the contour
        epsilon = max(0.01 * peri, 5)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4:  # Allow for more than 4 sides, as perspective might distort the shape
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / float(h)

            if aspectRatioRange[0] <= aspectRatio <= aspectRatioRange[1] and area >= minArea and area > largest_area:
                largest_area = area
                largest_square = approx

    return largest_square

# Main Code
webCamFeed = True
pathImage = "1.jpg"
cap = cv2.VideoCapture(0)  # Use RTSP feed
cap.set(10, 160)
heightImg = 480
widthImg = 480

# Initialize trackbars with initial values for Threshold1, Threshold2, and Kernel Size
initializeTrackbars([120, 180, 3])  # The third value is the initial kernel size
count = 0
while True:
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    # Applying CLAHE to increase the contrast of the grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgGray = clahe.apply(imgGray)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    thresVal1, thresVal2, kernelSize = valTrackbars() 
    #thres = valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    #imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    imgThreshold = cv2.Canny(imgBlur, thresVal1, thresVal2)
    #kernel = np.ones((5, 5))
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    #imgClose = cv2.morphologyEx(imgThreshold, cv2.MORPH_CLOSE, kernel, iterations=3)
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    # FIND ALL CONTOURS
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # FIND THE LARGEST SQUARE
    largest_square = findLargestSquare(contours)
    imgSquares = img.copy()
    if largest_square is not None:
        cv2.drawContours(imgSquares, [largest_square], -1, (0, 255, 0), 2)

    # Image Array for Display
    imageArray = ([img, imgSquares],
                  [imgThreshold, imgThreshold])  # We no longer need imgContours

    # LABELS FOR DISPLAY
    labels = [["Original", "Squares"],
              ["Threshold", "Threshold"]]

    stackedImage = stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)

    # CHECK FOR 'q' KEY TO QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# RELEASE THE WEBCAM AND DESTROY ALL OPENED CV2 WINDOWS
cap.release()
cv2.destroyAllWindows()
