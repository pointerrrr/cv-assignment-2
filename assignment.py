import glm
import random
import numpy as np
import cv2 as cv
import calibrate

block_size = 1.0

checkerBoardWidth = 8
checkerBoardHeight = 6
checkerBoardSquareSize = 115

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
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    return data


def get_cam_positions():
    cam1pos = rotated_cam_pos("cam1")
    cam2pos = rotated_cam_pos("cam2")
    cam3pos = rotated_cam_pos("cam3")
    cam4pos = rotated_cam_pos("cam4")

    cam_positions = [cam1pos, cam2pos, cam3pos, cam4pos]
    # shuffle coordinates to rotate positions properly
    return map(lambda x: [x[0], x[2] * -1, x[1]], cam_positions)


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam1rot = rotated_cam_rot("cam1")
    cam2rot = rotated_cam_rot("cam2")
    cam3rot = rotated_cam_rot("cam3")
    cam4rot = rotated_cam_rot("cam4")
    cam_angles = [cam1rot, cam2rot, cam3rot, cam4rot]
    return cam_angles

def rotated_cam_pos(camName):
    rvec = get_cam_rot(camName)
    tvec = get_cam_pos(camName)
    tvec /= checkerBoardSquareSize
    rotation_matrix = cv.Rodrigues(rvec)[0]
    return -np.matrix(rotation_matrix).T * np.matrix(tvec)

def get_cam_pos(camName):
    return np.load('data/' + camName + '/camTvec.npy')

def get_cam_rot(camName):
    return np.load('data/' + camName + '/camRvec.npy')

def rotated_cam_rot(camName):
    swap_vector = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]])
    rotateMat = get_cam_rot(camName)

    r = cv.Rodrigues(rotateMat)[0]

    rotation = np.zeros((4, 4), dtype=np.float32)
    #rotation[3][3] = 1.0
    rotation[:3, :3] = r

    matrix = np.matmul(rotation, swap_vector)
    return glm.mat4(*matrix.flatten().tolist())
