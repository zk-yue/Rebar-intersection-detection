import numpy as np
import open3d as o3d
from sympy import *
import math
# b = np.array([-2,-3,0,0,0,6,4,1])
# print(b==0)
# c=np.array([0,1,2,3,4,5,6,7])
# print(c[b==0])
# print( np.arange(0,10))
# for i in range(1):
#     print("进入循环")

# t = symbols('t')
# result = list(solveset(Eq(2*t+1, 3*t+3), t))
# print(type(result))
# print(result)
# cross_point_x=A_save[3][0]*result[0]+B_save[3][0]
# cross_point_y=A_save[3][1]*result[0]+B_save[3][1]
# cross_point_z_1=A_save[3][2]*result[0]+B_save[3][2]
# cross_point_z_2=A_save[3][2]*result[0]+B_save[3][2]

# t_1, t_2= symbols('t_1, t_2')
# result=solve([3*t_1-4*t_2-1,t_1-t_2-5], [t_1,t_2])
# print(result)

A_save=np.load("A.npy")
B_save=np.load("B.npy")
index_class_0=np.load("index_class_0.npy")
index_class_1=np.load("index_class_1.npy")
# print(A_save)
# print(B_save)
# print(index_class_0)
# print(index_class_1)

# t_1, t_2 = symbols('t_1 t_2')
# # for m in range(index_class_0.shape[0]):
# #     for n in range(index_class_1.shape[0]):
# m=0
# n=3
# result_t= solve([A_save[m][0]*t_1+B_save[m][0]-(A_save[n][0]*t_2+B_save[n][0]),A_save[m][1]*t_1+B_save[m][1]-(A_save[n][1]*t_2+B_save[n][1])],[t_1, t_2])
# print(type(result_t))
# print(result_t)
# print(result_t[t_2])

arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02,
                                               cone_radius=0.05,
                                               cylinder_height=0.3,
                                               cone_height=0.1,
                                               resolution=20,
                                               cylinder_split=4,
                                               cone_split=1)
arrow.compute_vertex_normals()
arrow.paint_uniform_color([1, 0, 0])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
cloud_show=[]
cloud_show.append(mesh_frame)
cloud_show.append(arrow)

reference_vector = np.array([0, 0, 1])
for m in range(index_class_0.shape[0]):
    for n in range(index_class_1.shape[0]):
        vector_1=A_save[index_class_0[m]]
        vector_2=A_save[index_class_1[n]]
        perpendicular_vector = np.cross(vector_1, vector_2)
        if perpendicular_vector[2]<0:
            perpendicular_vector=-perpendicular_vector
        # 创建一个单位旋转矩阵
        rotation_matrix = o3d.geometry.RotationMatrix()    
        # 计算将 vector1 旋转到与 vector2 方向一致的旋转矩阵
        rotation_matrix = rotation_matrix.get_rotation_matrix_from_two_vectors(reference_vector, perpendicular_vector)
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02,
                                               cone_radius=0.05,
                                               cylinder_height=0.3,
                                               cone_height=0.1,
                                               resolution=20,
                                               cylinder_split=4,
                                               cone_split=1)
        arrow.paint_uniform_color([0, 1, 0])
        cloud_show.append(arrow)
o3d.visualization.draw_geometries(cloud_show)
