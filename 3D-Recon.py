import torch
from opt import get_opts
import numpy as np
from einops import rearrange
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import time

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays
from models.networks import NGP
from models.rendering import render
from train import depth2img
from utils import load_ckpt

import warnings;

import os

import open3d as o3d

import queue

warnings.filterwarnings("ignore")


class OrbitCamera:
    def __init__(self, K, img_wh, r):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r
        self.center = np.zeros(3)
        self.rot = np.eye(3)

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(0.05 * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NGPGUI:
    def __init__(self):
        self.directory_choose = "/home/yangzesong/Projects/ngp_pl/data/yangang_Recon/flowerbed"

        # NeRF parameters
        self.num_epochs = 3
        self.downsample = 0.5
        self.scale = 1.0
        self.save_folder = "flowerbed"

        self.whether_inference = False

        self.log_info = queue.Queue(maxsize=10)  # 只显示5条消息就够了

        self.register_dpg()

    def register_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title="四牌楼重建", width=256, height=256, resizable=True)  # 创建窗口

        # 初始化界面
        #   menu1:稀疏重建，并且弹出稀疏重建的windows
        #       1. 选取图片对应路径
        #       2. 在text中显示图片路径，并且设置一个button，是否进行稀疏重建
        #               如果确认，首先查找文件夹中是否存在sparse文件夹，存在则直接完成，否则调用colmap进行重建
        #       3. 完成稀疏重建后，进行提示

        #   menu2：NGP重建，弹出NGP的选项
        #       1. 选择NGP的选项 --scale --num_epochs --downsample --exp
        #       2. 然后os.system()运行训练代码，实时显示当前训练进度
        #       3. 训练完成后，提示训练完成
        #
        #   menu3：显示结果，弹出show的选项
        #       1. 输入需要显示的exp文件名
        #       2. 显示结果

        # **************************************************稀疏重建**************************************************
        def directory_choose(sender, app_data, user_data):
            # 首先设置input_text显示选择的路径
            if dpg.does_item_exist("directory_path"):
                dpg.set_value("directory_path", app_data["file_path_name"])
            self.directory_choose = app_data["file_path_name"]
            print("The directory chosed where the images are is ", self.directory_choose)

        def showPointCloud():
            # 首先判断是否有ply文件
            if not os.path.exists(os.path.join(self.directory_choose, "sparse/test.ply")):
                # 如果没有ply文件，那么先进行转换
                cmd2 = f"colmap model_converter \
                    --input_path {self.directory_choose}/sparse/0/ \
                    --output_path {self.directory_choose}/sparse/test.ply \
                    --output_type PLY"
                os.system(cmd2)

            dpg.add_text(default_value="Find the PLY file",
                         parent="Sparse_Reconstruction")
            # 用open3d展示点云
            dpg.add_text(default_value="Use +/- to scale the point size",
                         parent="Sparse_Reconstruction")

            path = os.path.join(self.directory_choose, "sparse/test.ply")
            pcd = o3d.io.read_point_cloud(path, format="ply")
            o3d.visualization.draw_geometries(
                [pcd],
                zoom=0.3412,
                front=[0.4257, -0.2125, -0.8795],
                lookat=[2.6172, 2.0475, 1.532],
                up=[-0.0694, -0.9768, 0.2024]
            )

        def show_info(parent):
            for i in range(self.log_info.qsize()):
                if dpg.does_item_exist(f"show_info_{i}"):
                    # 如果该text栏已经存在，判断是否属于当前窗口
                    if dpg.get_item_parent(f"show_info_{i}") == parent:  # 如果当前窗口对应的item已经存在，那不需要再创建，直接赋值即可，否则将其删除
                        dpg.set_value(f"show_info_{i}", self.log_info.queue[i])
                    else:
                        # 如果当前窗口并不存在对应的item，则删除item，重新创建一个属于当前窗口的item
                        dpg.delete_item(f"show_info_{i}")
                        dpg.add_text(default_value=self.log_info.queue[i], parent=parent, tag=f"show_info_{i}")
                        print(f"在{parent}窗口中创建了show_info_{i} text栏")
                else:
                    # 如果并不存在该text栏，则创建一个
                    dpg.add_text(default_value=self.log_info.queue[i], parent=parent, tag=f"show_info_{i}")
                    print(f"在{parent}窗口中创建了show_info_{i} text栏")

        def Colmap_GetPose():
            dpg.add_progress_bar(label="", tag="Sparse_bar", default_value=0.0, parent="Sparse_Reconstruction")

            if os.path.exists(os.path.join(self.directory_choose, "sparse")):
                # 如果本地已经有sparse文件夹，说明已经完成稀疏重建
                dpg.add_text(default_value="There already has the sparse file, Start training straight away",
                             parent="Sparse_Reconstruction")
                dpg.set_value("Sparse_bar", 1.0)
                # dpg.set_item_label("Sparse_bar", "Finish the sparse reconstruction"),
                dpg.add_text(default_value="Finish the sparse reconstruction", parent="Sparse_Reconstruction")
            else:
                data_dir = self.directory_choose

                dpg.add_text(default_value="Feature Extracting...", parent="Sparse_Reconstruction", tag="show_stage")
                print("———————特征提取———————")
                cmd1 = f"colmap feature_extractor \
                                --database_path {data_dir}/database.db \
                                --image_path {data_dir}/images/ \
                                --ImageReader.camera_model PINHOLE \
                                --SiftExtraction.gpu_index 0"
                extracting_info = os.popen(cmd1)
                # 使用一个长度为10的队列来记录终端输出，并且实时显示在UI中
                for info in extracting_info:
                    if self.log_info.full():
                        self.log_info.get()
                        self.log_info.put(info)
                    else:
                        self.log_info.put(info)
                    show_info(parent="Sparse_Reconstruction")

                dpg.set_value("Sparse_bar", 0.25)

                dpg.set_value("show_stage", "Feature Matching...")
                print("———————特征匹配———————")
                cmd2 = f"colmap exhaustive_matcher \
                                --database_path {data_dir}/database.db \
                                --SiftMatching.gpu_index 0"

                matching_info = os.popen(cmd2)
                for info in matching_info:
                    if self.log_info.full():
                        self.log_info.get()
                        self.log_info.put(info)
                    else:
                        self.log_info.put(info)
                    show_info(parent="Sparse_Reconstruction")
                dpg.set_value("Sparse_bar", 0.5)

                dpg.set_value("show_stage", "Solving the poses...")
                print("———————位姿求解———————")
                os.makedirs(f"{data_dir}/sparse/0", exist_ok=True)
                cmd3 = f"colmap mapper \
                                --database_path {data_dir}/database.db \
                                --image_path {data_dir}/images \
                                --output_path {data_dir}/sparse "

                pose_info = os.popen(cmd3)
                for info in pose_info:
                    if self.log_info.full():
                        self.log_info.get()
                        self.log_info.put(info)
                    else:
                        self.log_info.put(info)
                    show_info(parent="Sparse_Reconstruction")

                dpg.set_value("Sparse_bar", 1.0)
                dpg.set_value("show_stage", "Finish the sparse reconstruction...")

            dpg.add_button(label="show the sparse pointclouds", tag="showPointClouds", callback=showPointCloud,
                           parent="Sparse_Reconstruction")
            dpg.add_button(label="NeRF training", callback=NeRF_Reconstruction, parent="Sparse_Reconstruction")

        def Sparse_Reconstruction():
            # 进行稀疏重建
            #   首先选择路径
            if dpg.does_item_exist("Sparse_Reconstruction"):  # 如果已经打开一个窗口了，就将其删掉
                dpg.delete_item("Sparse_Reconstruction")

            with dpg.window(label="Sparse_Reconstruction", tag="Sparse_Reconstruction", width=800, height=600):
                # 为选择文件夹的按钮设置风格
                if dpg.does_item_exist("directory_choose"):  # 如果已经打开一个窗口了，就将其删掉
                    dpg.delete_item("directory_choose")

                dpg.add_file_dialog(
                    directory_selector=True, show=False, callback=directory_choose, user_data=self.directory_choose,
                    tag="directory_choose")

                # 选择文件夹页面
                dpg.add_button(label="Directory Selector", callback=lambda: dpg.show_item("directory_choose"),
                               parent="Sparse_Reconstruction")
                dpg.add_text("Choose a directory which consists of a image folder where the images of the scene in")
                dpg.add_input_text(label="The directory path chosed", tag="directory_path", width=500,
                                   parent="Sparse_Reconstruction")
                # 进行稀疏重建
                dpg.add_button(label="Sparse Reconstruction", callback=Colmap_GetPose, parent="Sparse_Reconstruction")
                dpg.add_separator()

        # **************************************************稀疏重建**************************************************

        # **************************************************NeRF训练**************************************************
        def NeRF_training():
            # python train.py --root_dir data/yangang_Recon/flowerbed --num_epochs 5 --downsample 0.5 --scale 1.0 --exp colmap
            '''
            self.scale = dpg.get_value("scale")
            self.downsample = dpg.get_value("downsample")
            self.num_epochs = dpg.get_value("num_epochs")
            self.save_folder = dpg.get_value("save_folder")
            '''
            '''
            不用训练的情况：已经有对应的ckpt
            要训练的情况：1. 有对应文件夹，但是没有对应的
            '''
            print(f"ckpts/colmap/{self.save_folder}/epoch={self.num_epochs - 1}_scale={int(100 * self.scale)}_downsample={int(100 * self.downsample)}_slim.ckpt")
            
            if os.path.exists(
                    f"ckpts/colmap/{self.save_folder}/epoch={self.num_epochs - 1}_scale={int(100 * self.scale)}_downsample={int(100 * self.downsample)}_slim.ckpt"):
                print("已经有现成的训练好的")
            else:
                cmd_train = f"python train.py --root_dir {self.directory_choose} " \
                            f"--num_epochs {self.num_epochs} " \
                            f"--downsample {self.downsample} " \
                            f"--scale {self.scale} " \
                            f"--exp {self.save_folder}"
                extracting_info = os.popen(cmd_train)
                for info in extracting_info:
                    if self.log_info.full():
                        self.log_info.get()
                        self.log_info.put(info)
                    else:
                        self.log_info.put(info)
                    show_info(parent="NeRF_Reconstruction")

            dpg.add_text(default_value="Finish the training...", parent="NeRF_Reconstruction")
            dpg.add_separator(parent="NeRF_Reconstruction")
            dpg.add_button(label="Inference", callback=Show_Render, parent="NeRF_Reconstruction")

        def NeRF_Reconstruction():
            #   menu2：NGP重建，弹出NGP的选项
            #       1. 选择NGP的选项 --scale --num_epochs --downsample --exp
            #       2. 然后os.system()运行训练代码，实时显示当前训练进度
            #       2. 然后os.system()运行训练代码，实时显示当前训练进度
            #       3. 训练完成后，提示训练完成
            if dpg.does_item_exist("NeRF_Reconstruction"):  # 如果已经打开一个窗口了，就将其删掉
                dpg.delete_item("NeRF_Reconstruction")

            with dpg.window(label="NeRF_Reconstruction", tag="NeRF_Reconstruction", height=800, width=1000):
                dpg.add_text(default_value="The image folder chosed to reconstruction is " + self.directory_choose)
                dpg.add_text(
                    default_value="The folder must have the sparse files,if not, please return to the Sparse Reconstruction!")

                if not dpg.does_item_exist("directory_choose"):  # 如果没有初始化过directory choose
                    dpg.add_file_dialog(directory_selector=True, show=False, callback=directory_choose,
                                        user_data=self.directory_choose,
                                        tag="directory_choose")
                dpg.add_button(label="Change folder", callback=lambda: dpg.show_item("directory_choose"))  # 更改folder
                dpg.add_separator()

                dpg.add_text(
                    default_value="Please set the parameters needed for training: ")
                dpg.add_text(
                    default_value="----scale:      The ratio of the entire scene display, generally set to 0.5 ~ 1.5,")
                dpg.add_text(
                    default_value="                set too large may lead to cuda out of memory ")
                dpg.add_text(
                    default_value="----downsample: Scaling factor of image resolution, generally set to 0.0 ~ 1.0,")
                dpg.add_text(
                    default_value="                if the image is too large you can reduce the resolution to speed up the training")
                dpg.add_text(
                    default_value="----num_epochs: The num of the training epochs, generally value of 2 the result is good,")
                dpg.add_text(
                    default_value="                larger value with better the effect, but relativly longer the training time")
                dpg.add_text(
                    default_value="----savefolder: Where the trained model will be saved")

                def set_scale(sender):
                    self.scale = dpg.get_value(sender)
                    print("scale is ", self.scale)

                def set_downsample(sender):
                    self.downsample = dpg.get_value(sender)
                    print("downsample is ", self.downsample)

                def set_epochs(sender):
                    self.num_epochs = dpg.get_value(sender)

                def set_savefolder(sender):
                    self.save_folder = dpg.get_value(sender)

                dpg.add_input_float(label="Scale of the scene", default_value=1.0, tag="scale", width=100,
                                    callback=set_scale)
                dpg.add_input_float(label="Downsample of the image", default_value=0.5, tag="downsample", step=0.05,
                                    callback=set_downsample, width=100)
                dpg.add_input_int(label="Num of epoch for trainning", default_value=3, tag="num_epochs",
                                  callback=set_epochs, width=100)
                dpg.add_input_text(label="Name of the folder to save results", default_value="flowerbed", width=100,
                                   callback=set_savefolder, tag="save_folder")

                dpg.add_button(label="NeRF Training", callback=NeRF_training)

        # **************************************************NeRF训练**************************************************

        # **************************************************可视化**************************************************
        def render_cam(cam):
            t = time.time()
            directions = get_ray_directions(cam.H, cam.W, cam.K, device='cuda')
            rays_o, rays_d = get_rays(directions, torch.cuda.FloatTensor(cam.pose))

            exp_step_factor = 1 / 256

            results = render(self.model, rays_o, rays_d,
                             **{'test_time': True,
                                'to_cpu': True, 'to_numpy': True,
                                'T_threshold': 1e-2,
                                'exposure': torch.cuda.FloatTensor([dpg.get_value('_exposure')]),
                                'max_samples': 100,
                                'exp_step_factor': exp_step_factor})

            rgb = rearrange(results["rgb"], "(h w) c -> h w c", h=self.H)
            depth = rearrange(results["depth"], "(h w) -> h w", h=self.H)
            torch.cuda.synchronize()
            self.dt = time.time() - t
            self.mean_samples = results['total_samples'] / len(rays_o)

            if self.img_mode == 0:
                return rgb
            elif self.img_mode == 1:
                return depth2img(depth).astype(np.float32) / 255.0

        def render_DPG():
            if dpg.does_item_exist("_texture"):
                dpg.set_value("_texture", render_cam(self.cam))
                dpg.set_value("_log_time", f'Render time: {1000 * self.dt:.2f} ms')
                dpg.set_value("_samples_per_ray", f'Samples/ray: {self.mean_samples:.2f}')

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("show_gui"):
                return
            self.cam.orbit(app_data[1], app_data[2])
            # render_DPG()

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("show_gui"):
                return
            self.cam.scale(app_data)
            # render_DPG()

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("show_gui"):
                return
            self.cam.pan(app_data[1], app_data[2])
            # render_DPG()

        def callback_depth(sender, app_data):
            self.img_mode = 1 - self.img_mode

        def Show_Render():
            # 首先进行初始化
            kwargs = {'root_dir': self.directory_choose,
                      'downsample': self.downsample,
                      'read_meta': False}
            dataset = dataset_dict["colmap"](**kwargs)

            rgb_act = "Sigmoid"
            self.model = NGP(scale=self.scale, rgb_act=rgb_act).cuda()
            self.ckpt_path = os.path.join(f"./ckpts/colmap/{self.save_folder}",
                                          f"epoch={self.num_epochs - 1}_scale={int(100 * self.scale)}_downsample={int(100 * self.downsample)}_slim.ckpt")

            print(self.ckpt_path)
            load_ckpt(self.model, self.ckpt_path)

            self.cam = OrbitCamera(dataset.K, dataset.img_wh, r=2.5)
            self.W, self.H = dataset.img_wh
            self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

            self.dt = 0
            self.mean_samples = 0
            self.img_mode = 0

            self.whether_inference = True

            if dpg.does_item_exist("show_gui"):
                print("已经存在show_gui了")
                dpg.delete_item("show_gui")

            if dpg.does_item_exist("_control_window"):
                dpg.delete_item("_control_window")

            with dpg.window(tag="show_gui", width=self.W, height=self.H):
                if dpg.does_item_exist("_texture"):
                    dpg.delete_item("_texture")
                with dpg.texture_registry(show=False):
                    dpg.add_raw_texture(
                        self.W,
                        self.H,
                        self.render_buffer,  # 需要显示的图片
                        format=dpg.mvFormat_Float_rgb,
                        tag="_texture")
                dpg.add_image("_texture")

            # dpg.set_primary_window("show_gui", True)  # 设置为主窗口

            # 控制窗口
            with dpg.window(label="Control", tag="_control_window", width=200, height=150):
                dpg.add_slider_float(label="exposure", default_value=0.2,
                                     min_value=1 / 60, max_value=32, tag="_exposure")
                dpg.add_button(label="show depth", tag="_button_depth",
                               callback=callback_depth)  # RGB与Depth进行切换
                dpg.add_separator()  # 分割线
                dpg.add_text('no data', tag="_log_time")  # 启动的时间
                dpg.add_text('no data', tag="_samples_per_ray")  # 每条光线的采样数

            # 负责检测鼠标动作的回调函数
            with dpg.handler_registry():  # 全局回调函数
                dpg.add_mouse_drag_handler(  # 检测到鼠标左键拖动，则旋转视野
                    button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
                )
                dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)  # 检测到滚轮滑动，改变视野远近
                dpg.add_mouse_drag_handler(
                    button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan  # 检测到滚轮按下并且移动，平移视野
                )

            ## Avoid scroll bar in the window ##
            with dpg.theme() as theme_no_padding:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_style(
                        dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                    )
                    dpg.add_theme_style(
                        dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                    )
                    dpg.add_theme_style(
                        dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                    )
            dpg.bind_item_theme("show_gui", theme_no_padding)

        # **************************************************可视化**************************************************

        #############1. 完成稀疏重建对应的menu
        with dpg.window(label="menu", tag="main_menu"):
            dpg.add_menu_item(label="Sparse Reconstruction", callback=Sparse_Reconstruction)
            dpg.add_separator()
            dpg.add_menu_item(label="NeRF Reconstruction", callback=NeRF_Reconstruction)
            dpg.add_separator()
            dpg.add_menu_item(label="Show the render result", callback=Show_Render)

        dpg.setup_dearpygui()
        # dpg.set_viewport_small_icon("assets/icon.png")
        # dpg.set_viewport_large_icon("assets/icon.png")

        # dpg.show_item_registry()

        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            if not dpg.does_item_exist("show_gui"):
                self.whether_inference = False
            if self.whether_inference:
                render_DPG()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


if __name__ == "__main__":
    NGPGUI()

# python show_gui.py --root_dir data/yangang_Recon/flowerbed --ckpt_path ckpts/colmap/exp/epoch=2_slim.ckpt
