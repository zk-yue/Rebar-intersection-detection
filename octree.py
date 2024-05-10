import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 加载点云
print("->正在加载点云... ")
dataset="point_cloud_00000.pcd"
pcd = o3d.io.read_point_cloud(dataset) #点云读取
print(pcd)

pcd.paint_uniform_color([1.0,0.0,0.0])
octree= o3d.geometry.Octree(max_depth=4)
octree.convert_from_point_cloud(pcd,size_expand=0.01)
o3d.visualization.draw_geometries([octree], 
                                  window_name = '八叉树'
                                #   width = 600,height = 450,
                                #   left = 30,top = 30,
                                )