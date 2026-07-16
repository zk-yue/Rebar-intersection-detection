import numpy as np
import open3d as o3d
from sympy import *

A_save=np.load("A.npy")
B_save=np.load("B.npy")
index_class_0=np.load("index_class_0.npy")
index_class_1=np.load("index_class_1.npy")
print(A_save)
print(B_save)
print(index_class_0)
print(index_class_1)

point_cloud_line_all = []
for i in range(index_class_0.shape[0]):
    # 假设参数方程为 P = (1, 2, 3) + t * (0.2, 0.5, 0.8)
    P0 = B_save[index_class_0[i]]
    direction = A_save[index_class_0[i]]
    t_values = np.linspace(-0.8, 0.8, num=100)  # 选择一些参数值

    # 计算直线上的点
    line_points = [P0 + t * direction for t in t_values]

    # 创建点云对象
    point_cloud_line = o3d.geometry.PointCloud()
    point_cloud_line.points = o3d.utility.Vector3dVector(line_points)
    line_color = np.random.uniform(0, 1, (1, 3))       # 直线点云随机赋色
    point_cloud_line.paint_uniform_color([line_color[:, 0], line_color[:, 1], line_color[:, 2]])
    point_cloud_line_all.append(point_cloud_line)

for i in range(index_class_1.shape[0]):
    # 假设参数方程为 P = (1, 2, 3) + t * (0.2, 0.5, 0.8)
    P0 = B_save[index_class_1[i]]
    direction = A_save[index_class_1[i]]
    t_values = np.linspace(-0.8, 0.8, num=100)  # 选择一些参数值

    # 计算直线上的点
    line_points = [P0 + t * direction for t in t_values]
    # print(type(line_points))
    # 创建点云对象
    point_cloud_line = o3d.geometry.PointCloud()
    point_cloud_line.points = o3d.utility.Vector3dVector(line_points)
    line_color = np.random.uniform(0, 1, (1, 3))       # 直线点云随机赋色
    point_cloud_line.paint_uniform_color([line_color[:, 0], line_color[:, 1], line_color[:, 2]])
    point_cloud_line_all.append(point_cloud_line)

# o3d.visualization.draw_geometries(point_cloud_line_all)

cross_point = []
# x = a_1*t+b_1
# y = a_2*t+b_2
# z = a_3*t+b_3
t_1, t_2 = symbols('t_1 t_2')
cross_point_temp=[]
cross_cnt=0
for m in range(index_class_0.shape[0]):
    for n in range(index_class_1.shape[0]):
        result= solve([A_save[index_class_0[m]][0]*t_1+B_save[index_class_0[m]][0]-(A_save[index_class_1[n]][0]*t_2+B_save[index_class_1[n]][0]),A_save[index_class_0[m]][1]*t_1+B_save[index_class_0[m]][1]-(A_save[index_class_1[n]][1]*t_2+B_save[index_class_1[n]][1])],[t_1, t_2])
        # print(result)
        cross_point_x=A_save[index_class_0[m]][0]*result[t_1]+B_save[index_class_0[m]][0]
        cross_point_y=A_save[index_class_0[m]][1]*result[t_1]+B_save[index_class_0[m]][1]
        cross_point_z_1=A_save[index_class_0[m]][2]*result[t_1]+B_save[index_class_0[m]][2]
        cross_point_z_2=A_save[index_class_1[n]][2]*result[t_2]+B_save[index_class_1[n]][2]
        cross_point_z_mean=(cross_point_z_2+cross_point_z_1)/2
        cross_point_temp.append(np.array([cross_point_x,cross_point_y,cross_point_z_mean]))
        cross_cnt=cross_cnt+1
print(cross_cnt)
print(cross_point_temp)
cross_point_set = np.array(cross_point_temp)
print(cross_point_set)
# cross_point_temp = np.random.rand(100, 3)
print('type(cross_point_temp):',type(cross_point_temp))
point_cloud_cross_point = o3d.geometry.PointCloud()
point_cloud_cross_point.points = o3d.utility.Vector3dVector(cross_point_set)
cross_point_color = np.array([1,0,0])     # 直线点云随机赋色
point_cloud_cross_point.paint_uniform_color([cross_point_color[0], cross_point_color[1], cross_point_color[2]])
point_cloud_line_all.append(point_cloud_cross_point)

grasp_point_mark=[]
for i in range(cross_point_set.shape[0]):
    # 创建球体网格
    sphere =  o3d.geometry.TriangleMesh.create_sphere(radius=0.0125, resolution=100)
    sphere.paint_uniform_color([1, 0, 0])
    # 平移球体到指定位置
    translation = cross_point_set[i]
    sphere.translate(translation)
    # 创建一个空点云对象，用于显示球体和其他几何对象
    grasp_point_mark.append(sphere)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
point_cloud_line_all.append(mesh_frame)
cloud_show=[]
cloud_show.extend(point_cloud_line_all)
cloud_show.extend(grasp_point_mark)

o3d.visualization.draw_geometries(cloud_show)

