#!/usr/bin/env python3

from sys import argv, exit

from timeit import default_timer as timer

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
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 300, 0.01)
            cv.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)
            corners_set.append(corners)

        return corners_set


def calibrate_pinhole(points3d, points2d, size):
    t1 = timer()

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 300, 0.1)
    rms, K, D, _, _ = cv.calibrateCamera(
        points3d, points2d, size, None, None, flags=cv.CALIB_RATIONAL_MODEL+cv.CALIB_TILTED_MODEL+cv.CALIB_THIN_PRISM_MODEL)

    return rms, K, D, timer() - t1


def calibrate_fisheye(points3d, points2d, size):
    t1 = timer()

    K = np.empty((3, 3))
    D = np.empty(4)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 3000, 0.1)
    rms, K, D, _, _ = cv.fisheye.calibrate(
        points3d, points2d, size, K, D, criteria=criteria, flags=cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC)

    return rms, K, D, timer() - t1


def calibrate_omnidir(points3d, points2d, size):
    t1 = timer()

    K = np.empty((3, 3))
    xi = None  # np.empty((0))
    D = None  # np.empty((0))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
    rms, K, xi, D, _, _, _ = cv.omnidir.calibrate(
        points3d, points2d, size, K, xi, D, 0, criteria)
    return rms, K, D, timer() - t1


def print_calibration(caption, rms, K, D, t):
    print()
    print(caption)
    print(f'rms: {rms}')
    print(f'K: {K}')
    print(f'D: {D}')
    print(f'time: {t * 1000} ms')


if __name__ == '__main__':

    # print(cv.getBuildInformation())

    if len(argv) < 2:
        print(f'usage: {argv[0]} image_list')
        exit()

    with open(argv[1]) as f:
        imageList = f.read().splitlines()

    pattern_size = (6, 8)
    image_size = (1032, 778)

    points2d = detect_corners(imageList)

    points3d = len(points2d) * [generate_pattern_points3d(pattern_size)]

    rms, K, D, t = calibrate_pinhole(points3d, points2d, image_size)
    print_calibration('pinhole', rms, K, D, t)

    rms, K, D, t = calibrate_fisheye(points3d, points2d, image_size)
    print_calibration('fisheye', rms, K, D, t)

    rms, K, D, t = calibrate_omnidir(points3d, points2d, image_size)
    print_calibration('omnidir', rms, K, D, t)
