

from objloader import OBJ


class Model():

    def __init__(self, path, aruco_id, swapyz=False, scale=[0, 0, 0], translation=[0, 0, 0]):
        self.obj = OBJ(path, swapyz=swapyz)
        self.translation = translation
        self.scale = scale
        self.aruco_id = aruco_id

    def render(self):
        self.obj.render()
