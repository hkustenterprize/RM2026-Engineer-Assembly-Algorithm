import mujoco
import numpy as np
import glfw


class Camera:
    """ROS Camera Wrapper for MuJoCo Rendering"""

    def __init__(self, model, name, width, height, out: np.ndarray = None, **kwargs):
        self.model = model
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        self.width, self.height = width, height
        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.out = out or np.empty((height, width, 3), dtype=np.uint8)


class GLFW:
    def __init__(self, model, data, cameras: list[Camera] = [],
                 window_width=720,
                 window_height=540,
                 title="MuJoCo",
                 enable_visualization=True,
                 vsync=True,
                 **kwargs
                 ):
        self.model, self.data = model, data
        self.enable_visualization = enable_visualization
        self.cameras = cameras

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        offscreen_width, offscreen_height = kwargs.get(
            'offscreen_width', 720), kwargs.get('offscreen_height', 540)
        for cam in cameras:
            offscreen_width = max(offscreen_width, cam.width)
            offscreen_height = max(offscreen_height, cam.height)

        # Set offscreen buffer size to match the largest camera or context
        # This is critical for NVIDIA/Linux to get correct image sizes offscreen
        model.vis.global_.offwidth = max(
            model.vis.global_.offwidth, offscreen_width)
        model.vis.global_.offheight = max(
            model.vis.global_.offheight, offscreen_height)

        # Here we adopt the default behavior from glfw official implmementation
        # User could set the render backend via environment variable
        # i.e, __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
        glfw.default_window_hints()
        # Create windows
        if not enable_visualization:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.window = glfw.create_window(
            window_width, window_height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if vsync else 0)

        # Renderer context (binded to offscreen buffer)
        self.con = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)

        # mujoco.MjvOption: Control which elements are rendered
        # mujoco.MjvPerturb: Perturbations of the scene (e.g., for interactive dragging)
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()

        if enable_visualization:
            self.debug_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.debug_cam)
            self._setup_callbacks()

    def _setup_callbacks(self):
        self.last_mouse_x = self.last_mouse_y = 0
        self.button_left = self.button_right = False

        def scroll(w, x, y): self.debug_cam.distance = max(
            0.5, min(10.0, self.debug_cam.distance * (1 - 0.1 * y)))

        def mouse(w, b, a, m):
            self.button_left = (glfw.get_mouse_button(
                w, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
            self.button_right = (glfw.get_mouse_button(
                w, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
            self.last_mouse_x, self.last_mouse_y = glfw.get_cursor_pos(w)

        def cursor(w, x, y):
            dx, dy = x - self.last_mouse_x, y - self.last_mouse_y
            if self.button_left:
                self.debug_cam.azimuth -= dx * 0.3
                self.debug_cam.elevation = max(-90, min(0,
                                               self.debug_cam.elevation - dy * 0.3))
            elif self.button_right:
                self.debug_cam.lookat[0] -= dx * \
                    0.001 * self.debug_cam.distance
                self.debug_cam.lookat[1] += dy * \
                    0.001 * self.debug_cam.distance
            self.last_mouse_x, self.last_mouse_y = x, y
        glfw.set_scroll_callback(self.window, scroll)
        glfw.set_mouse_button_callback(self.window, mouse)
        glfw.set_cursor_pos_callback(self.window, cursor)

    def render_cameras(self, ros_cameras: list[Camera], data=None):
        """
        Render the scene to multiple ROS cameras using the offscreen buffer.
        """
        if not ros_cameras:
            return

        if self.enable_visualization:
            # Force to poll the events to keep the context alive
            glfw.poll_events()

        # glfw.make_context_current(self.window)
        d = data or self.data
        # Set to offscreen buffer, important for NVIDIA drivers
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        for c in ros_cameras:
            mujoco.mjv_updateScene(self.model, d, self.opt, self.pert,
                                   c.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            mujoco.mjr_render(c.rect, self.scene, self.con)
            mujoco.mjr_readPixels(c.out, None, c.rect, self.con)
            c.out[:] = np.flipud(c.out)
        # glfw.poll_events()

    def render_window(self, data=None):
        """
        Render the debug UI to the GLFW window.
        """
        if not self.enable_visualization:
            return True
        if glfw.window_should_close(self.window):
            return False

        d = data or self.data
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)
        w, h = glfw.get_framebuffer_size(self.window)
        mujoco.mjv_updateScene(self.model, d, self.opt, self.pert,
                               self.debug_cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self.scene, self.con)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return True

    def update(self, ros_cameras=[], data=None):
        """
        Combined update for both cameras and window.
        """
        self.render_cameras(ros_cameras, data)
        return self.render_window(data)

    def __del__(self):
        glfw.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        glfw.terminate()


if __name__ == "__main__":
    # 示例用法
    xml = """
    <mujoco>
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type='plane' size='2 2 .01' rgba='.9 .9 .9 1'/>
            <body name="origin" pos='0 0 0'/>
            <body name="ball" pos='0 0 1'>
                <freejoint/>
                <geom type='sphere' size='.1' rgba='1 0 0 1'/>
            </body>
            <!-- 定义两个相机 -->
            <camera name="cam_front" pos="1.5 0 1" mode="targetbody" target="origin"/>
            <camera name="cam_side" pos="0 0.5 2" mode="targetbody" target="ball"/>
        </worldbody>
    </mujoco>
    """
    # model = mujoco.MjModel.from_xml_string(xml)
    model = mujoco.MjSpec.from_string(xml).compile()
    data = mujoco.MjData(model)

    ros_cams = [Camera(model, "cam_front", 1440, 1080),
                Camera(model, "cam_side", 1440, 1080)]

    import time
    start = time.perf_counter()
    has_saved = False
    count = 0
    last_ = start
    mujoco.mj_step(model, data)
    import cv2

    with GLFW(model, data, window_width=720, window_height=540,
              cameras=ros_cams,
              enable_visualization=True, vsync=True) as win:

        while win.update(ros_cams):
            count += 1
            now = time.perf_counter()

            # img = ros_cams[0].out
            # img2 = ros_cams[1].out

            # # concat = np.concatenate((img, img2), axis=1)
            # cv2.imshow("ROS Camera View", img)
            # cv2.waitKey(1)

            # if has_saved:
            mujoco.mj_step(model, data)
            last = time.perf_counter()
            elapsed = last - start
            # print(f"Step time: {elapsed*1000:.4f} ms")
            start = last
            time.sleep(0.001)
