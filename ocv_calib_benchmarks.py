#!/usr/bin/env python3

import numpy as np
import cv2 as cv


def generate_pattern_points3d(size):
    pattern_points = np.zeros((np.prod(size), 3), np.float64)
    pattern_points[:, :2] = np.indices(size).T.reshape(-1, 2)
    pattern_points *= 1

    objp = np.zeros((1, size[0]*size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

    return objp


def detect_corners(image_list):
    corners_set = []
    for path in image_list:
        image = cv.imread(path, 0)
        status, corners = cv.findChessboardCorners(image, (6, 8))
        if status:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)
            corners_set.append(corners)

        return corners_set


def calibrate_pinhole(points3d, points2d, size):
    rms, K, D, _, _ = cv.calibrateCamera(
        points3d, points2d, size, None, None)

    return rms, K, D


def calibrate_fisheye(points3d, points2d, size):
    K = np.empty((3, 3))
    D = np.empty((4))

    rms, K, D, _, _ = cv.fisheye.calibrate(points3d, points2d, size, K, D)

    return rms, K, D


def print_calibration(caption, rms, K, D):
    print()
    print(caption)
    print(f'rms: {rms}')
    print(f'K: {K}')
    print(f'D: {D}')


if __name__ == '__main__':
    with open('list') as f:
        imageList = f.read().splitlines()

    pattern_size = (6, 8)

    points2d = detect_corners(imageList)

    points3d = len(points2d) * [generate_pattern_points3d(pattern_size)]
    print(points3d[0])

    rms, K, D = calibrate_pinhole(points3d, points2d, pattern_size)
    print_calibration('pinhole', rms, K, D)

    image_size = (1032, 778)
    rms, K, D = calibrate_fisheye(points3d, points2d, image_size)
    print_calibration('fisheye', rms, K, D)
