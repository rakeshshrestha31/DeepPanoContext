import os
import sys
import cv2
import numpy as np

# class Equirectangular:
#     def __init__(self, img):
#         self._img = img
#         self._height, self._width = self._img.shape[:2]
#         #cp = self._img.copy()
#         #w = self._width
#         #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
#         #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
#
#
#     def GetPerspective(self, FOV, THETA, PHI, height, width, RADIUS = 128):
#         #
#         # THETA is left/right angle, PHI is up/down angle, both in degree
#         #
#
#         equ_h = self._height
#         equ_w = self._width
#         equ_cx = (equ_w - 1) / 2.0
#         equ_cy = (equ_h - 1) / 2.0
#
#         wFOV = FOV
#         hFOV = float(height) / width * wFOV
#
#         c_x = (width - 1) / 2.0
#         c_y = (height - 1) / 2.0
#
#         wangle = (180 - wFOV) / 2.0
#         w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
#         w_interval = w_len / (width - 1)
#
#         hangle = (180 - hFOV) / 2.0
#         h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
#         h_interval = h_len / (height - 1)
#         x_map = np.zeros([height, width], np.float32) + RADIUS
#         y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
#         z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
#         D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
#         xyz = np.zeros([height, width, 3], np.float)
#         xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
#         xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
#         xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]
#
#         y_axis = np.array([0.0, 1.0, 0.0], np.float32)
#         z_axis = np.array([0.0, 0.0, 1.0], np.float32)
#         [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
#         [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))
#
#         xyz = xyz.reshape([height * width, 3]).T
#         xyz = np.dot(R1, xyz)
#         xyz = np.dot(R2, xyz).T
#
#         lat = np.arcsin(np.clip(xyz[:, 2] / RADIUS, -1., 1.))
#         lon = np.zeros([height * width], np.float)
#         theta = np.arctan(xyz[:, 1] / xyz[:, 0])
#         idx1 = xyz[:, 0] > 0
#         idx2 = xyz[:, 1] > 0
#
#         idx3 = ((1 - idx1) * idx2).astype(np.bool)
#         idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
#
#         lon[idx1] = theta[idx1]
#         lon[idx3] = theta[idx3] + np.pi
#         lon[idx4] = theta[idx4] - np.pi
#
#         lon = lon.reshape([height, width]) / np.pi * 180
#         lat = -lat.reshape([height, width]) / np.pi * 180
#         lon = lon / 180 * equ_cx + equ_cx
#         lat = lat / 90 * equ_cy + equ_cy
#         #for x in range(width):
#         #    for y in range(height):
#         #        cv2.circle(self._img, (int(lon[y, x]), int(lat[y, x])), 1, (0, 255, 0))
#         #return self._img
#
#         persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
#         return persp


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out

class Equirectangular:
    def __init__(self, img):
        self._img = img
        self._height, self._width = self._img.shape[:2]
        # [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp
