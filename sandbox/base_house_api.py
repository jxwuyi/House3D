import numpy as np

class BaseHouse:
    def __init__(self, resolution):
        self.obsMap = None
        self.moveMap = None
        pass

    def _setHouseBox(self, _lo, _hi, _rad):
        pass

    def _setObsMap(self, val):
        self.obsMap = val

    def _setMoveMap(self, val):
        self.moveMap = val

    def _genMovableMap(self, regions):
        pass

    def _genShortestDistMap(self, boxes, tag):
        pass

    def _genValidCoors(self, x1, y1, x2, y2, reg_tag):
        pass

    def _clearCurrentDistMap(self):
        pass

    def _setCurrentDistMap(self, tag):
        pass

    def _getConnMap(self):
        pass

    def _getInroomDist(self):
        pass

    def _getConnCoors(self):
        pass

    def _getMaxConnDist(self):
        pass

    def _getValidCoors(self, reg_tag):
        pass

    def _fetchValidCoorsSize(self, reg_tag):
        pass

    def _getCachedIndexedValidCoor(self, k):
        pass

    def _getConnectCoorsSize(self, tag):
        pass

    def _getConnectCoorsSize_Bounded(self, tag, bounded):
        pass

    def _getIndexedConnectCoor(self, tag, k):
        pass

    def _getCurrConnectCoorsSize(self):
        pass

    def _getCurrConnectCoorsSize_Bounded(self, bound):
        pass

    def _getCurrIndexedConnectCoor(self, k):
        pass

    def _inside(self, x, y):
        pass

    def _canMove(self, x, y):
        pass

    def _isConnect(self, x, y):
        pass

    def _getDist(self, x, y):
        pass

    def _getScaledDist(self, x, y):
        pass

    def _rescale(self, x1, y1, x2, y2, n_row):
        pass

    def _to_grid(self, x, y, n_row):
        pass

    def _to_coor(self, x, y, shft):
        pass

    def _check_occupy(self, cx, cy):
        pass

    def _full_collision_check(self, Ax, Ay, Bx, By, num_samples):
        pass

    def _fast_collision_check(self, Ax, Ay, Bx, By, num_samples):
        pass



