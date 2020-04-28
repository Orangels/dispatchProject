import time, os
import numpy as np
from threading import Thread


def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper


class Profiler(object):
    def __init__(self, names=['main']):
        self.names = names
        self.lasts = {k: 0 for k in names}
        self.totals = self.lasts.copy()
        self.counts = self.lasts.copy()
        self.means = self.lasts.copy()
        self.reset()

    def _insert(self, name):
        self.lasts[name] = time.time()
        self.totals[name] = 0
        self.counts[name] = 0
        self.means[name] = 0
        self.names.append(name)

    def reset(self):
        last = time.time()
        for name in self.names:
            self.lasts[name] = last
            self.totals[name] = 0
            self.counts[name] = 0
            self.means[name] = 0

    def start(self, name='main'):
        if name not in self.names:
            self._insert(name)
        self.lasts[name] = time.time()

    def stop(self, name='main'):
        if name not in self.names:
            self._insert(name)
        self.totals[name] += time.time() - self.lasts[name]
        self.counts[name] += 1
        self.means[name] = self.totals[name] / self.counts[name]

    def bump(self, name='main'):
        self.stop(name)
        self.start(name)


class recorder(object):
    def __init__(self, N=100):
        self.idx = 0
        self.N = N
        self.s = []
        self.c = []
        self.b = []
        self.tempfile = ''

    def save(self, score, box, cls):
        self.s.extend([score.copy()])
        self.c.extend([cls.copy()])
        self.b.extend([box.copy()])
        self.idx += 1
        self.remove_record()
        if self.idx % self.N == 0:
            np.savez('xxx_%d' % self.idx, s=self.s, c=self.c, b=self.b)
            self.s = []
            self.c = []
            self.b = []
            self.tempfile = ''
        else:
            np.savez('xxx_%d' % self.idx, s=self.s, c=self.c, b=self.b)
            self.tempfile = 'xxx_%d.npz' % self.idx

    def remove_record(self):
        if self.tempfile:
            os.system("rm %s" % self.tempfile)


def find_rect(points, shape, pad):
    p_array = np.array(points)
    x = p_array[:, 0]
    y = p_array[:, 1]

    xmax, xmin = np.max(x), np.min(x)
    ymax, ymin = np.max(y), np.min(y)
    xmin, xmax = xmin, xmax
    ymin, ymax = ymin, ymax

    if xmin < 0:
        xmin = 0
    if xmax >= shape[1]:
        xmax = shape[1]
    if ymin < 0:
        ymin = 0
    if ymax >= shape[0]:
        ymax = shape[0]
    rect_out = [[xmin + pad, ymin], [xmax + pad, ymax]]
    return rect_out


#########################################################################
# https://juejin.im/entry/56ca7c1e8ac2470053a3789d
# 点
class Point(object):

    def __init__(self, x, y):
        self.x, self.y = x, y


# 向量
class Vector(object):

    def __init__(self, start_point, end_point):
        self.start_point, self.end_point = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y


ZERO = 1e-9


def negative(vector):
    """取反"""
    return Vector(vector.end_point, vector.start_point)


def vector_product(vectorA, vectorB):
    '''计算 x_1 * y_2 - x_2 * y_1'''
    return vectorA.x * vectorB.y - vectorB.x * vectorA.y


def is_intersected(A, B, C, D):
    '''A, B, C, D 为 Point 类型'''
    AC = Vector(A, C)
    AD = Vector(A, D)
    BC = Vector(B, C)
    BD = Vector(B, D)
    CA = negative(AC)
    CB = negative(BC)
    DA = negative(AD)
    DB = negative(BD)
    return (vector_product(AC, AD) * vector_product(BC, BD) <= ZERO) \
           and (vector_product(CA, CB) * vector_product(DA, DB) <= ZERO)


def cross_line(aax, aay, bbx, bby, ccx, ccy, ddx, ddy):
    aa = Point(aax, aay)
    bb = Point(bbx, bby)
    cc = Point(ccx, ccy)
    dd = Point(ddx, ddy)
    return is_intersected(aa, bb, cc, dd)


#########################################################################

def find_direct(rect):
    x = int((rect[0][0] + rect[1][0]) / 2)
    y1 = rect[0][1] + int((rect[1][1] - rect[0][1]) / 4)
    y2 = rect[1][1] - int((rect[1][1] - rect[0][1]) / 4)
    direction_out = [[x, y1], [x, y2]]
    return direction_out


def generalequation(first_x, first_y, second_x, second_y):
    coeff_a = second_y - first_y
    coeff_b = first_x - second_x
    coeff_c = second_x * first_y - first_x * second_y
    return coeff_a, coeff_b, coeff_c


def cross_point(line1, line2):
    x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
    x3, y3, x4, y4 = line2[0], line2[1], line2[2], line2[3]
    coeff_a1, coeff_b1, coeff_c1 = generalequation(x1, y1, x2, y2)
    coeff_a2, coeff_b2, coeff_c2 = generalequation(x3, y3, x4, y4)
    m = coeff_a1 * coeff_b2 - coeff_a2 * coeff_b1
    if m == 0:
        x = None
        y = None
        return [x, y]
    else:
        x = (coeff_c2 * coeff_b1 - coeff_c1 * coeff_b2) / m
        y = (coeff_c1 * coeff_a2 - coeff_c2 * coeff_a1) / m
    return [int(x), int(y)]


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0] - 1
    n2 = x2.shape[0] - 1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.NaN

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]
