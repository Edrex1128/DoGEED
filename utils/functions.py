#!/user/bin/env python
# _*_ coding: utf-8 _*_
# ==================================================
# @File_name: DoGEED-main -> functions
# @Software: PyCharm
# @Author: 张福正
# @Time: 2025/1/21 20:21
# ==================================================

import numpy as np
import math

class switch(object):

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:
            self.fall = True
            return True
        else:
            return False

def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * s))
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel

def atan2(x):
    y = (2/math.pi) * np.sign(x) * np.arctan(x * x)
    return y


def translateImage(f, di, dj):
    N = f.shape[0]
    M = f.shape[1]
    if di > 0:
        iind = list(range(di, N))
        iind.append(N-1)
        # print(iind)
    elif di < 0:
        iind = list(range(N+di))
        iind.insert(0,0)
        # print(iind)
    else:
        iind = list(range(N))
        # print(iind)

    if dj > 0:
        jind = list(range(dj, M))
        jind.append(M-1)
        # print(jind)
    elif dj < 0:
        jind = list(range(M+dj))
        jind.insert(0,0)
        # print(jind)
    else:
        jind = list(range(M))
        # print(jind)

    ftrans = f[iind, :]
    ftrans = ftrans[:, jind]

    return ftrans

def snldStep(L, c):
    cpc = translateImage(c, 1, 0)
    cmc = translateImage(c, -1, 0)
    ccp = translateImage(c, 0, 1)
    ccm = translateImage(c, 0, -1)
    Lpc = translateImage(L, 1, 0)
    Lmc = translateImage(L, -1, 0)
    Lcp = translateImage(L, 0, 1)
    Lcm = translateImage(L, 0, -1)
    r = ((cpc + c) * (Lpc - L) - (c + cmc) * (L - Lmc) +
         (ccp + c) * (Lcp - L) - (c + ccm) * (L - Lcm)) / 2
    return r

def tnldStep(L, a, b, c):
    Lpc = translateImage(L, 1, 0)
    Lpp = translateImage(L, 1, 1)
    Lcp = translateImage(L, 0, 1)
    Lmp = translateImage(L, -1, 1)
    Lmc = translateImage(L, -1, 0)
    Lmm = translateImage(L, -1, -1)
    Lcm = translateImage(L, 0, -1)
    Lpm = translateImage(L, 1, -1)
    amc = translateImage(a, -1, 0)
    apc = translateImage(a, 1, 0)
    bmc = translateImage(b, -1, 0)
    bcm = translateImage(b, 0, -1)
    bpc = translateImage(b, 1, 0)
    bcp = translateImage(b, 0, 1)
    ccp = translateImage(c, 0, 1)
    ccm = translateImage(c, 0, -1)

    r = -1 / 4 * (bmc + bcp) * Lmp + 1 / 2 * (ccp + c) * Lcp + \
    1 / 4 * (bpc + bcp) * Lpp + \
    1 / 2 * (amc + a) * Lmc - \
    1 / 2 * (amc + 2 * a + apc + ccm + 2 * c + ccp) * L + \
    1 / 2 * (apc + a) * Lpc + \
    1 / 4 * (bmc + bcm) * Lmm + \
    1 / 2 * (ccm + c) * Lcm - \
    1 / 4 * (bpc + bcm) * Lpm

    return r

def EED(e, eta, stepsize):
    Gy, Gx = np.gradient(e)
    grad1 = Gx * Gx + Gy * Gy
    grad2 = np.sqrt(grad1)
    m = grad2.shape[0]
    n = grad2.shape[1]

    grad2_erf = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            grad2_erf[i, j] = math.erf((grad2[i, j] / eta) ** 2)

    c2 = np.ones((m, n)) - grad2_erf
    c1 = 0.2 * c2
    a = (c1 * Gx ** 2 + c2 * Gy ** 2) / (grad1 + np.spacing(1))
    b = (c2-c1) * Gx * Gy / (grad1 + np.spacing(1))
    c = (c1 * Gy ** 2 + c2 * Gx ** 2) / (grad1 + np.spacing(1))
    r = tnldStep(e, a, b, c)
    e = e + stepsize * r
    return e