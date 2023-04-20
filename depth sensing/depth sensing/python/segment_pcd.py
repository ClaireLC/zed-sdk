import open3d as o3d
import cv2
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import json

os.environ['DISPLAY'] = ':1'

def main(args):

    # Open img
    img_path = os.path.join(args.data_dir, "img_left.png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth_path = os.path.join(args.data_dir, "depth.npy")
    depth_arr = np.load(depth_path)

    intrinsics_path = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "cam_params.json")
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)

    # Get mask for box
    mask = get_mask(img, viz=True)
    idxs_y, idxs_x = mask.nonzero()

    depth_masked = depth_arr[idxs_y, idxs_x]

    seg_points = get_point_from_pixel(idxs_x, idxs_y, depth_masked, intrinsics["left"])
    seg_colors = img[idxs_y, idxs_x, :] / 255.0 # Normalize to [0,1] for cv2

    print(seg_points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seg_points)
    pcd.colors = o3d.utility.Vector3dVector(seg_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    #opt.background_color = np.asarray([0., 0., 0.])
    opt.background_color = np.asarray([0.2, 0.2, 0.2])
    vis.add_geometry(pcd)
    vis.update_renderer()
    vis.run()
    vis.destroy_window()



def get_point_from_pixel(px, py, depth, intrinsics):

    x = (px - intrinsics["cx"])/intrinsics["fx"]
    y = (py - intrinsics["cy"])/intrinsics["fy"]

    # For inverse brown conrady distortion?
    # From rs2_deproject_pixel_to_point() 
    # Coeffs of camera I'm using are all 0 anyway.
    #r2 = x*x + y*y
    #f = 1 + intrinsics.coeffs[0]*r2 + intrinsics.coeffs[1]*r2*r2 + intrinsics.coeffs[4]*r2*r2*r2
    #ux = x*f + 2*intrinsics.coeffs[2]*x*y + intrinsics.coeffs[3]*(r2 + 2*x*x)
    #uy = y*f + 2*intrinsics.coeffs[3]*x*y + intrinsics.coeffs[2]*(r2 + 2*y*y)
    #x = ux
    #y = uy

    X = x * depth
    Y = y * depth
    p = np.stack((X, Y, depth), axis=1)

    return p

def get_points(img):
    """
    Get pixels on image to generate mask for
    by clicking on image
    """
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    orig_img_bgr = img_bgr.copy()

    pt_list = []
    def click_and_get_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = [x, y]
            pt_list.append(pt)
            cv2.circle(img_bgr, (x,y), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_get_points)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", img_bgr)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            img_bgr = orig_img_bgr.copy()
            pt_list = []
        # if the "q" key is pressed, break from the loop
        elif key == ord("q"):
            break   
    cv2.destroyAllWindows()
    return np.array(pt_list)



def get_mask(img, viz=False):
    """
    Get mask for box

    Args:
        img: RGB image
    """

    # Load SAM model
    sam_ckpt = "/juno/u/clairech/zed-sdk/depth sensing/depth sensing/python/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device="cuda")

    # Choose points on object to segment
    input_points = get_points(img)
    input_labels = np.ones(input_points.shape[0])

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    score = scores[best_idx]

    if viz:
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        show_points(input_points, input_labels, plt.gca())
        plt.title(f"Mask {best_idx+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()  

    return mask


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", help="Data directory path")
    args = parser.parse_args()
    main(args)
