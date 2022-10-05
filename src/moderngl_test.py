# from pyrr import Matrix44
# import moderngl
# from PIL import Image

# class LoadingOBJ():
#     title = "Loading OBJ"
#     gl_version = (3, 3)

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         self.obj = self.load_scene('sitting_dummy.obj')
#         self.texture = self.load_texture_2d('wood.jpg')

#         self.prog = self.ctx.program(
#             vertex_shader='''
#                 #version 330
#                 uniform mat4 Mvp;
#                 in vec3 in_position;
#                 in vec3 in_normal;
#                 in vec2 in_texcoord_0;
#                 out vec3 v_vert;
#                 out vec3 v_norm;
#                 out vec2 v_text;
#                 void main() {
#                     v_vert = in_position;
#                     v_norm = in_normal;
#                     v_text = in_texcoord_0;
#                     gl_Position = Mvp * vec4(in_position, 1.0);
#                 }
#             ''',
#             fragment_shader='''
#                 #version 330
#                 uniform sampler2D Texture;
#                 uniform vec4 Color;
#                 uniform vec3 Light;
#                 in vec3 v_vert;
#                 in vec3 v_norm;
#                 in vec2 v_text;
#                 out vec4 f_color;
#                 void main() {
#                     float lum = dot(normalize(v_norm), normalize(v_vert - Light));
#                     lum = acos(lum) / 3.14159265;
#                     lum = clamp(lum, 0.0, 1.0);
#                     lum = lum * lum;
#                     lum = smoothstep(0.0, 1.0, lum);
#                     lum *= smoothstep(0.0, 80.0, v_vert.z) * 0.3 + 0.7;
#                     lum = lum * 0.8 + 0.2;
#                     vec3 color = texture(Texture, v_text).rgb;
#                     color = color * (1.0 - Color.a) + Color.rgb * Color.a;
#                     f_color = vec4(color * lum, 1.0);
#                 }
#             ''',
#         )

#         self.light = self.prog['Light']
#         self.color = self.prog['Color']
#         self.mvp = self.prog['Mvp']

#         # Create a vao from the first root node (attribs are auto mapped)
#         self.vao = self.obj.root_nodes[0].mesh.vao.instance(self.prog)

#     def render(self, time, frame_time):
#         self.ctx.clear(1.0, 1.0, 1.0)
#         self.ctx.enable(moderngl.DEPTH_TEST)

#         proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
#         lookat = Matrix44.look_at(
#             (-85, -180, 140),
#             (0.0, 0.0, 65.0),
#             (0.0, 0.0, 1.0),
#         )

#         self.light.value = (-140.0, -300.0, 350.0)
#         self.color.value = (1.0, 1.0, 1.0, 0.25)
#         self.mvp.write((proj * lookat).astype('f4'))

#         self.texture.use()
#         self.vao.render()

# if __name__ == '__main__':
#     LoadingOBJ.run()


import numpy as np
from PIL import Image
import moderngl
import moderngl_window
from pathlib import Path
import cv2 as cv

from moderngl_window.timers.clock import Timer
from moderngl_window.conf import settings
from moderngl_window.meta import (
    TextureDescription,
    ProgramDescription,
    SceneDescription,
    DataDescription,
)
from moderngl_window import resources

from pyrr import Matrix44
import moderngl_window as mglw


class HeadlessTest():
    """
    Simple one frame renderer writing to png and exit.
    If you need more fancy stuff, see the custom_config* examples.
    """
    samples = 0  # Headless is not always happy with multisampling
    window_size = (640, 480)

    def __init__(self):

        settings.WINDOW['class'] = 'moderngl_window.context.headless.Window'
        self.wnd = moderngl_window.create_window_from_settings()
        mglw.activate_context(window=self.wnd)
        self.wnd.name = "headless"
        self.ctx = self.wnd.ctx
        self.transform = Matrix44.identity()
        self.aspect_ratio = 640/480

        moderngl_window.resources.register_scene_dir(Path(__file__).parent)
        moderngl_window.resources.register_texture_dir(Path(__file__).parent)

        if self.wnd.name != 'headless':
            raise RuntimeError(
                'This example only works with --window headless option')

        self.obj = resources.scenes.load(
            SceneDescription(
                path="teapot.obj"
            )
        )
        self.texture = resources.textures.load(
            TextureDescription(
                path="wood.jpg"
            )
        )

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                }
            ''',
            fragment_shader='''
                #version 330

                out vec4 color;
                void main() {
                    color = vec4(0.04, 0.04, 0.04, 1.0);
                }
            ''',
        )

        # self.light = self.prog['Light']
        self.mvp = self.prog['Mvp']

        # Create a vao from the first root node (attribs are auto mapped)
        self.vao = self.obj.root_nodes[0].mesh.vao.instance(self.prog)

    def render(self):
        """Render one frame, save to png and close it"""
        self.ctx.clear(1.0, 1.0, 1.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        proj = Matrix44.perspective_projection(
            45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            (-85, -180, 140),
            (0.0, 0.0, 65.0),
            (0.0, 0.0, 1.0),
        )
        # print("xxxxxxxxxxxxxxxxxxxxxxx")
        # print(proj*lookat)
        # print("--------------")
        # print(Matrix44(self.transform))

        # self.light.value = (-140.0, -300.0, 350.0)
        # self.mvp.write((proj*lookat).astype('f4'))
        self.mvp.write(Matrix44(self.transform).astype('f4'))

        self.texture.use()
        self.vao.render()

        # Wait for all rendering calls to finish (Might not be needed)
        self.ctx.finish()

        image = Image.frombytes('RGBA', (640, 480),
                                self.wnd.fbo.read(components=4))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save('aruco.png', format='png')

        return image

        # self.wnd.close()

    def set_transform(self, transform):
        self.transform = transform

    def run(self):

        # self.wnd.clear()
        self.render()
        self.wnd.swap_buffers()

    def extrinsic2ModelView(self, RVEC, TVEC, R_vector=True):
        """[Get modelview matrix from RVEC and TVEC]
        Arguments:
            RVEC {[vector]} -- [Rotation vector]
            TVEC {[vector]} -- [Translation vector]
        """

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
        print(M)
        return M.T.flatten()

    def intrinsic2Project(self, MTX, width, height, near_plane=0.01, far_plane=1000.0):
        """[Get ]
        Arguments:
            MTX {[np.array]} -- [The camera instrinsic matrix that you get from calibrating your chessboard]
            width {[float]} -- [width of viewport]]
            height {[float]} -- [height of viewport]
        Keyword Arguments:
            near_plane {float} -- [near_plane] (default: {0.01})
            far_plane {float} -- [far plane] (default: {100.0})
        Returns:
            [np.array] -- [1 dim array of project matrix]
        """
        P = np.zeros(shape=(4, 4), dtype=np.float32)

        fx, fy = MTX[0, 0], MTX[1, 1]
        cx, cy = MTX[0, 2], MTX[1, 2]

        P[0, 0] = 2 * fx / width
        P[1, 1] = 2 * fy / height
        P[2, 0] = 1 - 2 * cx / width
        P[2, 1] = 2 * cy / height - 1
        P[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
        P[2, 3] = -1.0
        P[3, 2] = - (2 * far_plane * near_plane) / (far_plane - near_plane)

        return P.flatten()


if __name__ == '__main__':
    HeadlessTest.run()
