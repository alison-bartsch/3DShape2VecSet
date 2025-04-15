import time
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt

def animate_point_cloud(pcl, view='isometric', pltmap='viridis'):
    '''
    This function takes a point cloud and generates an animated gif with the point cloud
    rotating around the z-axis. The camera remains in a fixed observation pose.
    '''
    img_sequence = []
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    # setr the point size
    ro = vis.get_render_option()
    ro.point_size = 10.0

    # set the camera view
    ctr = vis.get_view_control()
    base_path = "/home/alison/Documents/GitHub/subgoal_diffusion/open3d_configs"
    if view == 'isometric':
        path = base_path + "/isometric_view.json"
    elif view == 'side':
        path = base_path + "/side_on_view.json"
    elif view == 'top':
        path = base_path + "/top_down_view.json"
    else:
        raise ValueError("Invalid view type. Choose 'isometric', 'side', or 'top'.")
    parameters = o3d.io.read_pinhole_camera_parameters(path)

    # add in the geometry
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(pcl)
    geometry.colors = o3d.utility.Vector3dVector(generate_colormap(pcl, pltmap))
    vis.add_geometry(geometry)

    img = vis.capture_screen_float_buffer()
    img_sequence.append(img)

    ctr.convert_from_pinhole_camera_parameters(parameters, True)

    for i in range(360):
        geometry.points = o3d.utility.Vector3dVector(pcl)
        geometry.colors = o3d.utility.Vector3dVector(generate_colormap(pcl, pltmap))

        # Rotate the point cloud around the z-axis
        R = geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(i)))
        geometry.rotate(R, center=np.mean(pcl, axis=0))
        time.sleep(0.05)
        
        
        # Update the visualizer
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()

        # Capture the screen
        img = vis.capture_screen_float_buffer()
        img_sequence.append(img)
    return img_sequence

def set_camera_to_orthographic(vis):
    # get current camra settings
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    # modify the intrinsic parameters to achieve orthographic projection
    param.intrinsic.set_intrinsics(
        width=1920,
        height=1080,
        fx=1.0,
        fy=1.0,
        cx=960,
        cy=540
    )
    ctr.convert_from_pinhole_camera_parameters(param)

def generate_colormap(pcl, pltmap='viridis'):
    '''
    This function takes a point cloud and generates a colormap based on the z-coordinate of each point.
    The colormap is then applied to the point cloud and visualized.
    '''
    # Normalize the z-coordinates to the range [0, 1]
    z = pcl[:, 2]
    z_min = np.min(z)
    z_max = np.max(z)
    z_normalized = (z - z_min) / (z_max - z_min)

    # Create a colormap
    colormap = plt.get_cmap(pltmap)
    colors = colormap(z_normalized)

    return colors[:, :3]  # Ignore the alpha channel

def vis_fov_point_cloud(pcl):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    # setr the point size
    ro = vis.get_render_option()
    ro.point_size = 10.0

    # set the camera view
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("/home/alison/Documents/GitHub/subgoal_diffusion/open3d_configs/isometric_view.json")

    # add in the geometry
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(pcl)
    geometry.colors = o3d.utility.Vector3dVector(generate_colormap(pcl))
    vis.add_geometry(geometry) #, reset_bounding_box=False)

    ctr.convert_from_pinhole_camera_parameters(parameters, True)

    vis.run()
    vis.destroy_window()

def make_gif(img_sequence, filename='point_cloud_animation.gif', duration=100):
    '''
    This function takes a list of images and creates a gif from them.
    '''
    images = []
    for img in img_sequence:
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        images.append(Image.fromarray(img))
    images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

if __name__ == "__main__":
    pcl = np.load('/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory3/unnormalized_pointcloud16.npy')
    img_list = animate_point_cloud(pcl, view='isometric', pltmap='spring')
    make_gif(img_list, filename='point_cloud_animation_test.gif', duration=100)
    # vis_fov_point_cloud(pcl)

# NOTE: we want to visualize from 2x or 3x views