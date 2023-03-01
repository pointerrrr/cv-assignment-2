import cv2 as cv
import numpy as np
from pathlib import Path

def main():
    avgBackground('cam1')
    avgBackground('cam2')
    avgBackground('cam3')
    avgBackground('cam4')
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

if __name__ == "__main__":
    main()