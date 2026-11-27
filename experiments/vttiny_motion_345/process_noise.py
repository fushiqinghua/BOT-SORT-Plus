"""Extra discrete random-acceleration covariance for a Lost track. Units are original RGB pixels."""
import numpy as np

ACCELERATION = np.zeros((8,2))
ACCELERATION[[0,1],[0,1]] = .5
ACCELERATION[[4,5],[0,1]] = 1.
UNIT_Q = ACCELERATION @ ACCELERATION.T


def features(mean):
    size = np.sqrt(max(mean[2]*mean[3], 1.))
    speed = np.linalg.norm(mean[4:6])
    return np.array([1., np.clip(np.log1p(5.*speed/size),0.,3.), np.clip(np.log(size/12.),-3.,3.)])


def variance(x, weights):
    return np.exp(np.clip(np.asarray(x) @ weights, -12., np.log(16.)))
