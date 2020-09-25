import numpy as np
import math


def c(a):  # cos angle in radiant
    return np.cos(a)


def s(a):  # sin angle in radiant
    return np.sin(a)


def R_x(a, mode="rad"):  # rotation matrix arround x
    return np.array([[1, 0, 0], [0, c(a), -s(a)], [0, s(a), c(a)]])


def R_y(a):  # rotation matrix around y
    return np.array([[c(a), 0, s(a)], [0, 1, 0], [-s(a), 0, c(a)]])


def R_z(a):  # rotation matrix around z
    return np.array([[c(a), -s(a), 0], [s(a), c(a), 0], [0, 0, 1]])


def Rxyz(alpha, beta, theta, mode="degree"):  # mitbewegt
    if mode == "degree":
        R = R_x(alpha * np.pi / 180).dot(R_y(beta * np.pi / 180).dot(R_z(theta * np.pi / 180)))
    elif mode == "rad":
        R = R_x(alpha).dot(R_y(beta).dot(R_z(theta)))
    else:
        R = None
    return R


def Rrpy(roll, pitch, yaw, mode="degree"):
    if mode == "degree":
        print("roll: {}, pitch: {}, yaw: {}".format(roll, pitch, yaw))
        R = R_y(yaw * np.pi / 180).dot(R_x(pitch * np.pi / 180).dot(R_z(roll * np.pi / 180)))
    elif mode == "rad":
        R = R_z(y).dot(R_y(p).dot(R_x(r)))
    else:
        R = None
    return R


def Rxzy(alpha, beta, theta, mode="degree"):
    if mode == "degree":
        R = R_x(theta * np.pi / 180).dot(R_z(beta * np.pi / 180).dot(R_y(alpha * np.pi / 180)))
    elif mode == "rad":
        R = R_x(theta).dot(R_z(beta).dot(R_y(alpha)))
    else:
        R = None
    return R


def Ryzx(alpha, beta, theta, mode="degree"):
    if mode == "degree":
        R = R_y(alpha * np.pi / 180).dot(R_z(beta * np.pi / 180).dot(R_x(theta * np.pi / 180)))
    elif mode == "rad":
        R = R_y(alpha).dot(R_z(beta).dot(R_x(theta)))
    else:
        R = None
    return R


def g2r(a):  # grad to radiant
    return a * np.pi / 180


def r2g(a):  # radiant to grad
    return a * 180 / np.pi


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi

def roationMatrixToYawAngle(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        z = 0
    return  z * 180 / np.pi
