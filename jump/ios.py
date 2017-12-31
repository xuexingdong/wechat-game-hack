import wda

from jump.game import WechatJump


class IOSJump(WechatJump):

    scale = 2

    def __init__(self):
        self.c = wda.Client('http://192.168.1.3:8100')
        self.s = self.c.session('com.tencent.xin')

    def tap(self, x, y):
        self.s.tap(x, y)

    def screenshot(self, path):
        self.c.screenshot(path)

    def swipe(self, x1, y1, x2, y2, duration):
        self.s.swipe(x1, y1, x2, y2, duration)

    def window_size(self):
        return self.s.window_size()


if __name__ == '__main__':
    jump = IOSJump()
    jump.go()
