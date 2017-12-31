import math
import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

ASSETS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/assets/'

threshold_chess_max = [130, 100, 100]
threshold_background_max = [221, 229, 255]
threshold_background_min = [204, 196, 200]


class WechatJump(ABC):
    # 屏幕截图像素与屏幕尺寸的比例，本比例为iPhone 6s
    scale = 2
    # 系数，距离 * 系数 = 按压时间
    ratio = 0.0023

    threshold_shadow_max = [150, 150, 180]
    threshold_shadow_min = [140, 140, 170]

    def go(self):
        print('成功加载微信')
        w, h = self.window_size()
        print('下拉寻找跳一跳入口')
        self.swipe(w / 2, 200, w / 2, 2 * h / 3, 0.5)
        print('点击logo')
        self.__click(ASSETS_PATH + 'logo.png')
        time.sleep(2)
        print('点击开始游戏')
        self.__click(ASSETS_PATH + 'start.png')
        time.sleep(1)
        self._jump()

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

    def _jump(self):
        # 判断游戏是否结束
        while self.__find(ASSETS_PATH + 'restart.png') is None:
            # 计算距离，设置
            dis = self._cal_dis()
            print('距离 %s 按压时间 %s' % (dis, dis * self.ratio))
            # 长按
            self.swipe(100, 100, 100, 100, dis * self.ratio)
            time.sleep(1)
        else:
            print('重新开始')
            self.__screenshot()
            self.__click(ASSETS_PATH + 'restart.png')
            time.sleep(1)
            self._jump()

    def _cal_dis(self):
        self.__screenshot()
        pic = cv2.imread('screenshot.png', cv2.IMREAD_COLOR)
        h, w, _ = pic.shape
        # 取中间1/3区域进行计算
        pic = pic[math.floor(h / 3):math.floor(h * 2 / 3), :]
        for row_idx, row in enumerate(pic):
            for col_idx, col in enumerate(row):
                # 过滤背景
                if threshold_background_min[0] <= col[0] <= threshold_background_max[0] \
                        and threshold_background_min[1] <= col[1] <= threshold_background_max[1] \
                        and threshold_background_min[2] <= col[2] <= threshold_background_max[2]:
                    pic[row_idx, col_idx] = np.array([0, 0, 0])
        # 根据灰度数组找出棋子位置
        chess_x, chess_y = self.__find_chess(pic)
        print('棋子位置', chess_x, chess_y)
        # 根据灰度数组找出下一个盒子的位置
        box_x, box_y = self.__find_next_box(pic)
        print('盒子位置', box_x, box_y)
        dis = abs(box_x - chess_x)
        return dis

    def __find_chess(self, sub_pic):
        gray_arr = cv2.cvtColor(sub_pic, cv2.COLOR_BGR2GRAY)
        gray_arr[np.logical_or(gray_arr < 50, gray_arr > 100)] = 255
        # 中值滤波去噪
        gray_arr = cv2.medianBlur(gray_arr, 5)
        _, binary = cv2.threshold(gray_arr, 80, 255, cv2.THRESH_BINARY)
        index_arr = np.where(binary == 0)
        # 找到最左侧的点
        left = index_arr[1].min()
        # 找到最右侧的点
        right = index_arr[1].max()
        # 找到最下方的点
        bottom = index_arr[0].max()
        x = (left + right) / 2
        # 圆底半径
        r = (right - left) / 2
        y = bottom - r
        return x, y

    def __find_next_box(self, sub_pic):
        gray_arr = cv2.cvtColor(sub_pic, cv2.COLOR_BGR2GRAY)
        # 找到出现次数最多的灰度值
        bg_gray = 206
        # 把该值当做背景色进行去除
        bool_index = np.logical_and(gray_arr >= bg_gray - 5, gray_arr <= bg_gray + 5)
        gray_arr[bool_index] = 0
        gray_arr = cv2.medianBlur(gray_arr, 5)
        index_arr = np.where(gray_arr != 0)
        top_y = index_arr[0].min()
        top_x = np.where(gray_arr[top_y] != 0)[0][0]
        right_x = index_arr[1].max()
        right_y = np.where(gray_arr[:, right_x] != 0)[0][0]
        return top_x, right_y

    def __screenshot(self):
        self.screenshot('screenshot.png')

    def __click(self, img):
        area: np.ndarray = self.__find(img)
        summary = area.sum(axis=0).sum(axis=0)
        self.tap(summary[0] / 4 / self.scale, summary[1] / 4 / self.scale)

    def __find(self, img):
        self.__screenshot()
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
        matches = bf.knnMatch(des1, des2, 2)
        # cv2.drawMatchesKnn expects list of lists as matches.
        # opencv3.0有drawMatchesKnn函数
        # Apply ratio test
        # 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
        # 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
        goods = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goods.append(m)
        if len(goods) <= 10:
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


def show(result):
    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 800, 480)
    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()
