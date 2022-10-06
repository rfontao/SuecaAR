import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import cv2 as cv
import numpy as np
from objloader import OBJ

model = ""

image = None
extrinsic = None
intrinsic = None


def initOpengl(width, height, pos_x=500, pos_y=500, window_name=b'Aruco Demo'):

    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(pos_x, pos_y)

    window_id = glutCreateWindow(window_name)
    # glutHideWindow()
    glutDisplayFunc(draw_scene)
    # glutIdleFunc(draw_scene)

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glShadeModel(GL_SMOOTH)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)

    # # Assign texture
    glEnable(GL_TEXTURE_2D)

    # Set ambient lighting
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1))


def draw_objects(image, intrinsic, extrinsic):
    global model

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMultMatrixf(intrinsic)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glLoadMatrixf(extrinsic)
    glScaled(0.001, 0.001, 0.001)
    # glTranslatef(self.translate_x, self.translate_y, self.translate_y)
    glCallList(model.gl_list)


# def draw_scene(image, intrinsic, extrinsic):
def draw_scene():
    global image, intrinsic, extrinsic
    print(image)
    print("DRAWING")
    if image != None:
        print("have image")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        image = draw_background(image)  # draw background
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_objects(image, intrinsic, extrinsic)  # draw the 3D objects.

        glutSwapBuffers()
        glutPostRedisplay()

        return image


def load_model(path):
    global model
    model = OBJ(path, swapyz=True)


def draw_background(image):

    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Setting background image project_matrix and model_matrix.
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(33.7, 1.3, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Convert image to OpenGL texture format
    bg_image = cv.flip(image, 0)
    bg_image = Image.fromarray(bg_image)
    ix = bg_image.size[0]
    iy = bg_image.size[1]
    bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

    # Create background texture
    texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, bg_image)

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

    # def run(self):
    #     # Begin to render
    #     glutMainLoop()