import glm
import random
import numpy as np
import cv2 as cv
import calibrate

block_size = 1.0

checkerBoardWidth = 8
checkerBoardHeight = 6
checkerBoardSquareSize = 115


class Camera:
    def __init__(self, camName, mat, dist, tvec, rvec, h, s, v):
        self.camName = camName
        self.mat = mat
        self.dist = dist
        self.tvec = tvec
        self.rvec = rvec
        self.h = h
        self.s = s
        self.v = v


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    for x in range(-18, 22):
        for y in range(0, 25):
            for z in range(-20, 20):
                data.append([x*block_size, y*block_size, z*block_size])


    data = np.array(data)
    camNames = ["cam1", "cam2", "cam3", "cam4"]
    cameras = []
    coords = []
    lookupTable = {}
    for camName in camNames:
        mat = load_cam_mat(camName)
        dist = load_cam_dist(camName)
        rvec = load_cam_rot(camName)
        tvec = load_cam_pos(camName)
        h = load_cam_h(camName)
        s = load_cam_s(camName)
        v = load_cam_v(camName)
        cam = Camera(camName, mat, dist, tvec, rvec, h , s, v)
        cameras.append(cam)

    projections = []

    for cam in cameras:
        background = cv.imread('data/' + camName + '/background_avg.jpg')
        vid = cv.VideoCapture('data/' + cam.camName + '/preMask.jpg', "video.avi")
        frame = vid.read()

        cv.imshow(get_foreground(frame, background, cam))
        cv.waitKey(5000)

    for cam in cameras:
        projections.append(cv.projectPoints(data, cam.rvec, cam.tvec, cam.mat, cam.dist)[0])

    for vidx, voxel in enumerate(data):
        voxelCoord = (voxel[0],voxel[1],voxel[2])
        lookupTable[voxelCoord] = []
        for cidx, cam in enumerate(cameras):
            lookupTable[voxelCoord].append(projections[cidx][vidx])

    return data

def get_foreground(frame, background, cam):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_background = cv.cvtColor(background, cv.COLOR_BGR2HSV)

    frame_channels = cv.split(hsv_frame)
    background_channels = cv.split(hsv_background)

    tmp = cv.absdiff(frame_channels[0], background_channels[0])
    _, foreground = cv.threshold(tmp, cam.h, 255, cv.THRESH_BINARY)

    tmp = cv.absdiff(frame_channels[1], background_channels[1])
    _, background = cv.threshold(tmp, cam.s, 255, cv.THRESH_BINARY)
    foreground = cv.bitwise_and(foreground, background)

    tmp = cv.absdiff(frame_channels[2], background_channels[2])
    _, background = cv.threshold(tmp, cam.v, 255, cv.THRESH_BINARY)
    foreground = cv.bitwise_or(foreground, background)

    kernel = np.ones((5, 5), np.uint8)

    foreground = cv.dilate(foreground, kernel)
    foreground = cv.erode(foreground, kernel)
    foreground = cv.erode(foreground, kernel)
    foreground = cv.dilate(foreground, kernel)

    return foreground

def get_cam_positions():
    cam1pos = get_cam_pos("cam1")
    cam2pos = get_cam_pos("cam2")
    cam3pos = get_cam_pos("cam3")
    cam4pos = get_cam_pos("cam4")

    cam_positions = [cam1pos, cam2pos, cam3pos, cam4pos]
    # shuffle coordinates to rotate positions properly
    cam_positions = map(lambda x: [x[0], x[2] * -1, x[1]], cam_positions)
    return cam_positions


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam1rot = get_cam_rot("cam1")
    cam2rot = get_cam_rot("cam2")
    cam3rot = get_cam_rot("cam3")
    cam4rot = get_cam_rot("cam4")
    return [cam1rot, cam2rot, cam3rot, cam4rot]


def get_cam_pos(camName):
    rvec = load_cam_rot(camName)
    tvec = load_cam_pos(camName)
    tvec /= checkerBoardSquareSize
    rotation_matrix = cv.Rodrigues(rvec)[0]
    return -np.matrix(rotation_matrix).T * np.matrix(tvec)


def get_cam_rot(camName):
    rvec = load_cam_rot(camName)
    r = cv.Rodrigues(rvec)[0]

    r = np.pad(r, ((0, 1), (0, 1)))

    rot = glm.mat4(r)

    return glm.rotate(rot, glm.radians(90), (0, 0, 1))


def load_cam_pos(camName):
    return np.load('data/' + camName + '/camTvec.npy')


def load_cam_rot(camName):
    return np.load('data/' + camName + '/camRvec.npy')


def load_cam_dist(camName):
    return np.load('data/' + camName + '/camDist.npy')


def load_cam_mat(camName):
    return np.load('data/' + camName + '/camMat.npy')

def load_cam_h(camName):
    f = open('data/' + camName + '/hsv.txt', "x")
    x = f.readlines()
    return int(x[0])

def load_cam_s(camName):
    f = open('data/' + camName + '/hsv.txt', "x")
    x = f.readlines()
    return int(x[1])

def load_cam_v(camName):
    f = open('data/' + camName + '/hsv.txt', "x")
    x = f.readlines()
    return int(x[2])
