import win32api, win32com, win32con, win32gui
import pythoncom

class Mouse:
    # Module `Mouse`

    MApi = lambda flag: (win32api.mouse_event(flag, *win32api.GetCursorPos(), 0, 0))

    @staticmethod
    def left_up():
        Mouse.MApi(win32con.MOUSEEVENTF_LEFTUP)

    @staticmethod
    def left_down():
        Mouse.MApi(win32con.MOUSEEVENTF_LEFTDOWN)

    @staticmethod
    def right_up():
        Mouse.MApi(win32con.MOUSEEVENTF_RIGHTUP)

    @staticmethod
    def right_down():
        Mouse.MApi(win32con.MOUSEEVENTF_RIGHTDOWN)


def get_loc(app_name='BlueStacks App Player'):
    pid = win32gui.FindWindow(None, app_name)
    left, top, right, bottom = win32gui.GetWindowRect(pid)
    return left, top, right, bottom
