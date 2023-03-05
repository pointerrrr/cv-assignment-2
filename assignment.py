import glm
import random
import numpy as np
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
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cam1pos = get_cam_pos("cam1")
    cam2pos = get_cam_pos("cam2")
    cam3pos = get_cam_pos("cam3")
    cam4pos = get_cam_pos("cam4")
    return [cam1pos / 100, cam2pos / 100, cam3pos / 100, cam4pos / 100]
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam1rot = get_cam_rot("cam1")
    cam2rot = get_cam_rot("cam2")
    cam3rot = get_cam_rot("cam3")
    cam4rot = get_cam_rot("cam4")
    #cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_angles = [cam1rot * 180 / np.pi, cam2rot * 180/np.pi, cam3rot * 180 / np.pi, cam4rot / 180 * np.pi]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations

def get_cam_pos(camName):
    return np.load('data/' + camName + '/camTvec.npy')

def get_cam_rot(camName):
    return np.load('data/' + camName + '/camRvec.npy')
