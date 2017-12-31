import time
from abc import ABC, abstractmethod

import cv2
import os

import numpy as np

ASSETS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/assets/'


class WechatJump(ABC):
    # 屏幕截图像素与屏幕尺寸的比例，本比例为iPhone 6s
    scale = 2

    def go(self):
        print('成功加载微信')
        w, h = self.window_size()
        print('下拉寻找跳一跳入口')
        self.swipe(w / 2, 100, w / 2, h / 2, 0.5)
        print('开始游戏')
        self.start()
        time.sleep(1)
        self.jump()

    @abstractmethod
    def window_size(self):
        """
        获取窗口大小
        :return: (width, height)
        """
        return 0, 0

    @abstractmethod
    def screenshot(self, path):
        pass

    @abstractmethod
    def swipe(self, x1, y1, x2, y2, duration):
        pass

    @abstractmethod
    def tap(self, x, y):
        pass

    def start(self):
        self.__click(ASSETS_PATH + 'logo.png')
        time.sleep(2)
        self.__click(ASSETS_PATH + 'start.png')

    def is_gameover(self):
        return self.__find(ASSETS_PATH + 'restart.png') is not None

    def restart(self):
        self.__click(ASSETS_PATH + 'restart.png')

    def jump(self):
        while not self.is_gameover():
            print('点击跳跃')
            self.swipe(100, 100, 100, 100, 2)
            time.sleep(2)
        else:
            print('重新开始')
            self.restart()
            time.sleep(1)
            self.jump()

    def __click(self, img):
        area: np.ndarray = self.__find(img)
        summary = area.sum(axis=0).sum(axis=0)
        self.tap(summary[0] / 4 / self.scale, summary[1] / 4 / self.scale)

    def __find(self, img):
        self.screenshot('screenshot.png')
        return WechatJump.find_area(img, 'screenshot.png')

    @staticmethod
    def find_area(one, another):
        """
        copy from internet
        :param one: source pic path
        :param another: target pic path
        :return: None when no match found or (x1, y1, x2, y2) area when matches
        """
        source = cv2.imread(one, 0)
        target = cv2.imread(another, 0)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(source, None)
        kp2, des2 = sift.detectAndCompute(target, None)
        # 蛮力匹配算法,有两个参数，距离度量(L2(default),L1)，是否交叉匹配(默认false)
        bf = cv2.BFMatcher()
        # 返回k个最佳匹配
        matches = bf.knnMatch(des1, des2, k=2)
        # cv2.drawMatchesKnn expects list of lists as matches.
        # opencv3.0有drawMatchesKnn函数
        # Apply ratio test
        # 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
        # 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
        goods = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goods.append(m)
        if len(goods) < 10:
            return None
        src_pts = np.reshape(np.float32([kp1[m.queryIdx].pt for m in goods]), (-1, 1, 2))
        dst_pts = np.reshape(np.float32([kp2[m.trainIdx].pt for m in goods]), (-1, 1, 2))
        h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if h is None:
            return None
        height, width = source.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换,获得在目标图像上对应的坐标。
        pts = np.reshape(np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]), (-1, 1, 2))
        dst = cv2.perspectiveTransform(pts, h)
        return dst


if __name__ == '__main__':
    jump = WechatJump()
    jump.go()
