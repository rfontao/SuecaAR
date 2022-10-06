import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import cv2 as cv
import numpy as np
from objloader import OBJ

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
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1))

        # OpenCV intrinsic
        self.projection = OpenGLRenderer.intrinsic2Projection(camera_matrix, width, height)
        
        self.rvec = None
        self.tvec = None
        self.image = None
        self.model = None

    def load_model(self, path):
        self.model = OBJ(path, swapyz=True)

    def draw_object(self):
        model_view = OpenGLRenderer.extrinsic2ModelView(self.rvec, self.tvec)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(self.projection)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(model_view)

        glScaled(0.01, 0.01, 0.01)
        # glTranslatef(self.translate_x, self.translate_y, self.translate_y)
        self.model.render()



    def draw_scene(self):
        self.draw_background()  # Draw Webcam image
        self.draw_object()  # Draw objects
        glutSwapBuffers()


    def draw_background(self):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setting background image project_matrix and model_matrix.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        img = self.image.copy()
        # Convert image to OpenGL texture format
        img = cv.flip(img, 0)
        img = Image.fromarray(img)
        width = img.size[0]
        height = img.size[1]
        img = img.tobytes("raw", "BGRX", 0, -1)

        # Create background texture
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, img)

        glTranslatef(0.0, 0.0, -10.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-4.0,  3.0, 0.0)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)

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
