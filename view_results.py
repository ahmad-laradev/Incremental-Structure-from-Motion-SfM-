import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud(r"Code\Results with Bundle Adjustment") # Adjust the path as needed

if pcd.is_empty():
    print("Failed to load point cloud")
else:    
    # Force color visualization
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Create new colored point cloud
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization window with custom settings
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add point cloud and set render options
    vis.add_geometry(colored_pcd)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 2.0  # Larger points
    opt.show_coordinate_frame = True
    
    # Set camera position
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.8)
    view_ctl.set_front([0.0, 0.0, -1.0])
    view_ctl.set_lookat([0.0, 0.0, 0.0])
    view_ctl.set_up([0.0, -1.0, 0.0])
    
# Adjust visualization parameters
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.get_render_option().point_size = 2.0  # Increase point size
vis.run()
vis.destroy_window()