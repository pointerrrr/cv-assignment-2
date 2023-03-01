import cv2 as cv
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
import lxml
import glob
import json

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
width = 8
height = 6
squaresize = 115

corners3 = []

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)

    def __mul__(self, other):
        return Point(self.x * other, self.y * other)


# click event for collecting corner points
def click_event(event, x, y, flags, params):
    global corners3
    if event == cv.EVENT_LBUTTONDOWN:
        corners3.append([x, y])

def startCamCalibration(camName):
    path = Path('data/' + camName + '/config.txt')
    if not path.is_file():
        cami = cv.VideoCapture('data/' + camName + '/intrinsics.avi')
        came = cv.VideoCapture('data/' + camName + '/checkerboard.avi')
        camMatrix, camDistortion, camrvec, camtvecs = calibrateCam(cami, came)
        f = open('data/' + camName + '/config.txt', "x")
        f.write(np.array2string(camMatrix) + "\n")
        f.write(np.array2string(camDistortion) + "\n")
        f.write(np.array2string(camrvec) + "\n")
        f.write(np.array2string(camtvecs) + "\n")

def main():
    startCamCalibration('cam1')
    startCamCalibration('cam2')
    startCamCalibration('cam3')
    startCamCalibration('cam4')

def drawAxis(frame, rvecs, tvecs, matrix, distortion):
    axis = np.float32([[3 * squaresize, 0, 0], [0, 3 * squaresize, 0], [0, 0, -3 * squaresize], [0, 0, 0]]).reshape(
        -1, 3)

    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, matrix, distortion)

    origin = tuple(map(int, (imgpts[3].ravel())))

    cv.line(frame, origin, tuple(map(int, (imgpts[0].ravel()))), (255, 0, 0), 3)
    cv.line(frame, origin, tuple(map(int, (imgpts[1].ravel()))), (0, 255, 0), 3)
    cv.line(frame, origin, tuple(map(int, (imgpts[2].ravel()))), (0, 0, 255), 3)

    return frame


def runCalibration(intrinsicFrames, extrinsicFrame):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    for index, obj in enumerate(objp):
        objp[index] = obj * squaresize

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = intrinsicFrames

    for frame in images:
        # Get Image
        if frame is None:
            break
        # Turn image to grayscale
        grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Try to find corners
        success, corners = cv.findChessboardCorners(grayImg, (width, height), None)

        # If successfully found
        if success:
            # Increase corner accuracy and append
            corners = cv.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners)
    ret, matrix, distortion, _, _ = cv.calibrateCamera(objpoints, imgpoints, grayImg.shape[::-1], None, None)

    checkerboard = extrinsicFrame
    cv.imshow('img', checkerboard)
    cv.setMouseCallback('img', click_event)
    while True:
        cv.imshow('img', checkerboard)
        cv.waitKey(1)

        if len(corners3) == 4:
            break

    boardPoints = []
    tR = Point(corners3[0][0], corners3[0][1])
    tL = Point(corners3[1][0], corners3[1][1])
    bR = Point(corners3[2][0], corners3[2][1])
    bL = Point(corners3[3][0], corners3[3][1])

    tRtobR = tR - bR
    tLtobL = tL - bL
    tRtobRd = tRtobR / (width - 1)
    tLtobLd = tLtobL / (width - 1)

    for i in range(height):
        for j in range(width):
            start = tR - tRtobRd * j
            end = tL - tLtobLd * j
            step = (end - start) / (height - 1)
            res = start + step * i
            boardPoints.append((res.x, res.y))

    objpoints.append(objp)
    boardPoints = list(map(lambda x: (np.float32(x[0]), np.float32(x[1])), boardPoints))
    corners4 = np.array(boardPoints).reshape(width * height, 1, 2)
    # print(corners4)
    # corners5 = cv.cornerSubPix(grayImg, corners4, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners4)

    # Draw and display the corners
    # for i in range(1,54):
    # cv.line(img, tuple(map(int, asdf[i-1])), tuple(map(int, asdf[i])), ((i%5) * 50, 255-(i%5) * 50, 0), 3)

    # print(corners4)
    #cv.drawChessboardCorners(checkerboard, (width, height), corners4, True)
    #cv.imshow('img', checkerboard)
    #cv.waitKey(5000)
    corners3.clear()

    #corners2 = cv.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), criteria)

    ret, rvecs, tvecs = cv.solvePnP(objp, corners4, matrix, distortion)

    finalImg = drawAxis(checkerboard, rvecs, tvecs, matrix, distortion)
    #frame = drawCube(checkerboard, rvecs, tvecs)

    cv.imshow('img', checkerboard)
    cv.waitKey(5000)
    return matrix, distortion, rvecs, tvecs

def calibrateCam(intrinsicsCam, extrinsicsCam):
    ret, frame = intrinsicsCam.read()
    frames = []
    while frame is not None:
        for i in range(50):
            ret, frame = intrinsicsCam.read()
        frames.append(frame)
    intrinsicsCam.release()
    _, extrinsicsFrame = extrinsicsCam.read()
    return runCalibration(frames, extrinsicsFrame)

if __name__ == "__main__":
    main()
