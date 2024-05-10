import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 加载点云
print("->正在加载点云... ")
dataset="point_cloud_00000.pcd"
pcd = o3d.io.read_point_cloud(dataset) #点云读取
print(pcd)

# 法线估计
radius = 0.01   # 搜索半径
max_nn = 30     # 邻域内用于估算法线的最大点数
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计




# 坐标轴显示
viewer = o3d.visualization.Visualizer()# 可视化句柄
viewer.create_window(window_name='可视化')
opt=viewer.get_render_option() # 界面参数选择
opt.background_color=np.asarray([1,1,1])# 背景颜色
opt.point_size = 1 # 点大小
opt.show_coordinate_frame=True# 添加坐标系

# # 颜色设置
# pcd.paint_uniform_color([0,0,1]) # 设置颜色
# color=np.array(pcd.colors)
# inlier=[i for i in range(0,color.shape[0]) if i%2==0]
# color[inlier]=[1,0,0]
# pcd.colors=o3d.utility.Vector3dVector(color[:,:])

viewer.add_geometry(pcd)
viewer.run()

# 点云可视化
print("->正在可视化点云")
o3d.visualization.draw_geometries([pcd], 
                                  window_name = 'cloud_raw',
                                #   width = 600,height = 450,
                                #   left = 30,top = 30,
                                  point_show_normal = True)

# # 隐藏点去除 
# # invert=True显示当前视角看不到的点
# _,pt_map=pcd.hidden_point_removal([0,0,0.25],25)
# pcd2=pcd.select_by_index(pt_map,invert=True)
# o3d.visualization.draw_geometries([pcd2])

# 深度图像读取
depth_raw = o3d.io.read_image('depth.png')
plt.subplot(1,1,1)
plt.title('depth')
plt.imshow(depth_raw)
plt.show()

# print("->正在DBSCAN聚类...")
# eps = 0.5           # 同一聚类中最大点间距
# min_points = 50     # 有效聚类的最小点数
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
# max_label = labels.max()    # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd],window_name='DBSCAN')

'''
print("->正在保存点云")
o3d.io.write_point_cloud("write.pcd", pcd, True)	# 默认false，保存为Binarty；True 保存为ASICC形式
print(pcd)
'''



# # RANSAC单平面分割
# '''
# import open3d as o3d

# # 加载点云
# print("->正在加载点云... ")
# dataset="point_cloud_00000.pcd"
# point_cloud = o3d.io.read_point_cloud(dataset) #点云读取
# print(point_cloud)

# distance_threshold = 0.01 # 距离阈值，判断是否属于平面模型
# ransac_n = 3 # 迭代次数
# num_iterations = 100 # 最大迭代次数

# # 设置RANSAC参数
# plane_model, inliers = o3d.geometry.PointCloud.segment_plane(
#     point_cloud,
#     distance_threshold=distance_threshold,
#     ransac_n=ransac_n,
#     num_iterations=num_iterations
# )

# # 从原始点云中提取属于平面模型的点
# inlier_cloud = point_cloud.select_by_index(inliers)

# print("->正在可视化点云")
# o3d.visualization.draw_geometries([inlier_cloud],window_name = 'cloud_raw')


# 点云转深度图
'''
tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(planes[min_index])
intrinsic = o3d.core.Tensor([[1606, 0, 320.1*3], [0, 539.2*3, 247.6*3],
                                 [0, 0, 1]])
 
# 3. PointCloud生成depth Image
depth_reproj = tensor_pcd.project_to_depth_image(width=2048,
                                          height=1536,
                                          intrinsics=intrinsic,
                                          depth_scale=5000.0,
                                          depth_max=10.0)

plt.subplot(1,1,1)
plt.title('depth')
plt.imshow(depth_reproj)
plt.show()
'''

# intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
#                                  [0, 0, 1]])
 
# # 3. PointCloud生成depth Image
# depth_reproj = planes[min_index].project_to_depth_image(width=2048,
#                                           height=1536,
#                                           intrinsics=intrinsic,
#                                           depth_scale=5000.0,
#                                           depth_max=10.0)

# fig, axs = plt.subplots(1, 1)
# axs[0].imshow(np.asarray(depth_reproj.to_legacy()))  # depth->ointCloud->depth
# plt.show()


# points = np.asarray(planes[0].points)

# # 输出前几个点的坐标
# num_points_to_print = 10
# for i in range(num_points_to_print):
#     print("Point {}: {}".format(i+1, points[i]))