from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import cv2 as cv
import numpy as np
from objloader import OBJ
from model import Model

# See https://github.com/BryceQing/OPENCV_AR/blob/master/AR_entrance.py


class OpenGLRenderer():

    def __init__(self, camera_matrix, width, height, pos_x=500, pos_y=500, window_name=b'ARSueca'):

        # Initializing GLUT Window
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(pos_x, pos_y)

        self.window_id = glutCreateWindow(window_name)
        glutDisplayFunc(self.draw_scene)
        glutIdleFunc(self.draw_scene)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)

        # Assign texture
        glEnable(GL_TEXTURE_2D)

        # Set ambient lighting
        glEnable(GL_LIGHTING)
        glLight(GL_LIGHT0, GL_POSITION,  (2, 2, 2, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 0, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.5, 0.5, 0.5, 1))
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

        # glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE | GL)

        # OpenCV intrinsic
        self.projection = OpenGLRenderer.intrinsic2Projection(
            camera_matrix, width, height)

        self.image = None
        self.models = []
        self.aruco_ids = []
        self.rvecs = []
        self.tvecs = []
        self.draw_model = False

    def draw_model(self, model, rvec, tvec):
        model_view = OpenGLRenderer.extrinsic2ModelView(rvec, tvec)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE | GL_SPECULAR)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(self.projection)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(model_view)

        glScaled(*model.scale)
        glTranslatef(*model.translation)
        model.render()

        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)

        # glDepthFunc(GL_LESS)

    def draw_scene(self):
        # Draw Webcam image
        self.draw_background()

        # Draw objects
        if self.draw_models:
            for i in range(len(self.aruco_ids)):
                for model in self.models:
                    if self.aruco_ids[i] == model.aruco_id:
                        self.draw_model(model, self.rvecs[i], self.tvecs[i])

        glutSwapBuffers()

    # See https://gist.github.com/Ashwinning/baeb0835624fedc2e5b809d42417b70e

    def draw_background(self):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)

        # Create background texture
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)

        img = self.image.copy()
        # Convert image to OpenGL texture format
        img = cv.flip(img, 0)
        img = Image.fromarray(img)
        width = img.size[0]
        height = img.size[1]
        img = img.tobytes("raw", "BGRX", 0, -1)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, width, height, 0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(0.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(width, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(width, height)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(0.0, height)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glClear(GL_DEPTH_BUFFER_BIT)

    def display(self):
        glutPostRedisplay()
        glutMainLoopEvent()

    @classmethod
    def extrinsic2ModelView(cls, RVEC, TVEC, R_vector=True):
        """Convert OpenCV extrinsic matrix to OpenGL Model View"""

        R, _ = cv.Rodrigues(RVEC)

        Rx = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

        TVEC = TVEC.flatten().reshape((3, 1))

        transform_matrix = Rx @ np.hstack((R, TVEC))
        M = np.eye(4)
        M[:3, :] = transform_matrix
        return M.T.flatten()

    @classmethod
    def intrinsic2Projection(cls, camera_matrix, width, height, near_plane=0.01, far_plane=100.0):
        """Convert OpenCV intrinsic matrix to OpenGL projection"""
        P = np.zeros(shape=(4, 4), dtype=np.float32)

        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        P[0, 0] = 2 * fx / width
        P[1, 1] = 2 * fy / height
        P[2, 0] = 1 - 2 * cx / width
        P[2, 1] = 2 * cy / height - 1
        P[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
        P[2, 3] = -1.0
        P[3, 2] = - (2 * far_plane * near_plane) / (far_plane - near_plane)

        return P.flatten()
