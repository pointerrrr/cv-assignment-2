import cv2 as cv
import numpy as np
from pathlib import Path

def main():
    avgBackground('cam1')
    avgBackground('cam2')
    avgBackground('cam3')
    avgBackground('cam4')

    subtractBackground('cam1')
    subtractBackground('cam2')
    subtractBackground('cam3')
    subtractBackground('cam4')

def avgBackground(camName):
    path = Path('data/' + camName + '/background.jpg')
    if not path.is_file():
        videocap = cv.VideoCapture('data/' + camName + '/background.avi')
        success, img = videocap.read()

        images = []
        while success:
            images.append(img)
            success, img = videocap.read()

        dst = images[0]
        for i in (range(len(images))):
            if i == 0:
                pass
            else:
                alpha = 1.0 / (i + 1)
                beta = 1.0 - alpha
                dst = cv.addWeighted(images[i], alpha, dst, beta, 0.0)

        cv.imwrite('data/' + camName + '/background_avg.jpg', dst)

def subtractBackground(camName):
    path = Path('data/' + camName + '/hsv.txt')
    if path.is_file():
       return
    thresholdH = 0
    thresholdS = 0
    thresholdV = 0


    frame = cv.imread('data/' + camName + '/preMask.jpg')
    _, mask = cv.threshold(cv.cvtColor(cv.imread('data/' + camName + '/mask.jpg'), cv.COLOR_BGR2GRAY), 127, 255, cv.THRESH_BINARY)
    background = cv.imread('data/' + camName + '/background_avg.jpg')

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_background = cv.cvtColor(background, cv.COLOR_BGR2HSV)

    frame_channels = cv.split(hsv_frame)
    background_channels = cv.split(hsv_background)

    min_dif = frame.shape[0] * frame.shape[1]

    for h in range(0,256,5):
        for s in range(0,256,5):
            for v in range(0,256,5):
                tmp = cv.absdiff(frame_channels[0], background_channels[0])
                _, foreground = cv.threshold(tmp, h, 255, cv.THRESH_BINARY)

                tmp = cv.absdiff(frame_channels[1], background_channels[1])
                _, background = cv.threshold(tmp, s, 255, cv.THRESH_BINARY)
                foreground = cv.bitwise_and(foreground, background)

                tmp = cv.absdiff(frame_channels[2], background_channels[2])
                _, background = cv.threshold(tmp, v, 255, cv.THRESH_BINARY)
                foreground = cv.bitwise_or(foreground, background)

                error = cv.bitwise_xor(mask, foreground)
                error = cv.countNonZero(error)

                if(error < min_dif):
                    min_dif = error
                    thresholdH = h
                    thresholdS = s
                    thresholdV = v

    for h in range(thresholdH, thresholdH + 5):
        for s in range(thresholdS - 5, thresholdS + 5):
            for v in range(thresholdV - 5, thresholdV + 5):
                tmp = cv.absdiff(frame_channels[0], background_channels[0])
                _, foreground = cv.threshold(tmp, h, 255, cv.THRESH_BINARY)

                tmp = cv.absdiff(frame_channels[1], background_channels[1])
                _, background = cv.threshold(tmp, s, 255, cv.THRESH_BINARY)
                foreground = cv.bitwise_and(foreground, background)

                tmp = cv.absdiff(frame_channels[2], background_channels[2])
                _, background = cv.threshold(tmp, v, 255, cv.THRESH_BINARY)
                foreground = cv.bitwise_or(foreground, background)

                error = cv.bitwise_xor(mask, foreground)
                error = cv.countNonZero(error)

                if(error < min_dif):
                    min_dif = error
                    thresholdH = h
                    thresholdS = s
                    thresholdV = v

    print('H: ' + str(thresholdH) + ' S: ' + str(thresholdS) + ' V: ' + str(thresholdV))
    f = open('data/' + camName + '/hsv.txt', "x")
    f.write(str(thresholdH) + "\n")
    f.write(str(thresholdS) + "\n")
    f.write(str(thresholdV) + "\n")
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    foreground = cv.erode(foreground, kernel)
    kernel = np.ones((2, 8), np.uint8)
    foreground = cv.dilate(foreground, kernel)
    kernel = np.reshape(kernel, (8, 2))
    foreground = cv.dilate(foreground, kernel)
    cv.imshow('img', foreground)
    cv.imwrite('data/' + camName + '/foreground.jpg', foreground)
    cv.waitKey(5000)

if __name__ == "__main__":
    main()