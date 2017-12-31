import os

from jump.game import WechatJump


class AndroidJump(WechatJump):
    def swipe(self, x1, y1, x2, y2, duration):
        cmd = 'adb shell input swipe %s %s %s %s %s' % (x1, y1, x2, y2, duration)
        os.system(cmd)

    def screenshot(self, path):
        os.system('adb shell screencap -p /sdcard/' + path)
        os.system('adb pull /sdcard/' + path + ' .')

    def window_size(self):
        # TODO
        pass

    def tap(self, x, y):
        cmd = 'adb shell input tap %s %s' % (x, y)
        os.system(cmd)
