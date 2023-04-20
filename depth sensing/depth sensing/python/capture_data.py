########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture a live 3D point cloud   
    with the ZED SDK and display the result in an OpenGL window.    
"""

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import os
import cv2
import numpy as np
import json
import argparse

os.environ['DISPLAY'] = ':1'


def save_intrinsics(camera_params, save_path=None):
    """ Save intrinsics of left and right camera in json """

    both_params = {
        "left": camera_params.left_cam,
        "right": camera_params.right_cam,
    }

    intrinsics = {}
    for cam_name, one_cam_params in both_params.items():
        intrinsics[cam_name] = {
            "fx": one_cam_params.fx,
            "fy": one_cam_params.fy,
            "cx": one_cam_params.cx,
            "cy": one_cam_params.cy,
            "disto": one_cam_params.disto.tolist(),
            "v_fov": one_cam_params.v_fov,
            "h_fov": one_cam_params.h_fov,
            "d_fov": one_cam_params.d_fov,
        }   

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(intrinsics, f, indent=4)

    return intrinsics
        



def main(args):
    
    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    )
    #init.depth_maximum_distance = 2 # Max distance in meters

    # Enable Positional tracking (mandatory for object detection)
    #positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    #positional_tracking_parameters.set_as_static = True

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 1280
    res.height = 720

    camera_model = zed.get_camera_information().camera_model
    camera_params = zed.get_camera_information().camera_configuration.calibration_parameters

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, camera_model, res)

    img_res_w = zed.get_camera_information().camera_resolution.width
    img_res_h = zed.get_camera_information().camera_resolution.height

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    img_left = sl.Mat(img_res_w, img_res_h, sl.MAT_TYPE.U8_C4)
    #quit()

    save_i = 0
    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, res)
            viewer.updateData(point_cloud)

            zed.retrieve_image(img_left, sl.VIEW.LEFT)
            img_left_arr = img_left.get_data()
            cv2.imshow("img_left", img_left_arr)
            cv2.waitKey(10)

            if(viewer.save_data == True):
    
                # Make save directory
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                    params_path = os.path.join(args.save_dir, "cam_params.json")
                    intrinsics = save_intrinsics(camera_params, save_path=params_path)
                data_dir = os.path.join(args.save_dir, f"data_{save_i}")
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
        
                # Save point cloud
                point_cloud_to_save = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU);
                zed.retrieve_measure(point_cloud_to_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                pcd_path = os.path.join(data_dir, f"pointcloud.ply")
                err = point_cloud_to_save.write(pcd_path)
                if(err == sl.ERROR_CODE.SUCCESS):
                    print("point cloud saved")
                else:
                    print("the point cloud has not been saved")

                # Save left and right rgb images
                img_left_to_save = sl.Mat(img_res_w, img_res_h, sl.MAT_TYPE.U8_C4);
                img_right_to_save = sl.Mat(img_res_w, img_res_h, sl.MAT_TYPE.U8_C4);
                zed.retrieve_image(img_left_to_save, sl.VIEW.LEFT, resolution=res)
                zed.retrieve_image(img_right_to_save, sl.VIEW.RIGHT, resolution=res)
                img_left_path = os.path.join(data_dir, f"img_left.png")
                img_right_path = os.path.join(data_dir, f"img_right.png")
                img_left_to_save.write(img_left_path)
                img_right_to_save.write(img_right_path)
    
                # Save depth image as array
                depth_img = sl.Mat(img_res_w, img_res_h, sl.MAT_TYPE.F32_C1)
                zed.retrieve_measure(depth_img, sl.MEASURE.DEPTH)
                depth_arr = depth_img.get_data()
                print(depth_arr.shape)
                depth_path = os.path.join(data_dir, f"depth.npy")
                np.save(depth_path, depth_arr)
            
                save_i += 1   
                viewer.save_data = False

    viewer.exit()
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-d", default="test", help="Directory to save images to")
    args = parser.parse_args()
    main(args)
