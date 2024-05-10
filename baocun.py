import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
from sklearn.cluster import KMeans
from sympy import *

# 计算相似度（使用余弦相似度）
def cosine_similarity(line1, line2):
    dot_product = np.dot(line1, line2)
    norm1 = np.linalg.norm(line1)
    norm2 = np.linalg.norm(line2)
    return dot_product / (norm1 * norm2)

# 1.加载点云
print("->正在加载点云... ")
dataset="point_cloud_00000.pcd"
point_cloud = o3d.io.read_point_cloud(dataset) #点云读取
print(point_cloud)

# 2.最上层钢筋点云分离
# 2.1 RANSAC算法参数设置
distance_threshold = 0.03 # 0.25 更好 0.3 更容易看出效果
ransac_n = 3
num_iterations = 100
# 2.2 RANSAC算法依次拟合各个平面
planes = [] # 用于保存平面中的点
planes_model_params = [] # 用于保存平面参数
while len(point_cloud.points) > 0:
    plane_model, inliers = o3d.geometry.PointCloud.segment_plane(
        point_cloud,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    inlier_cloud = point_cloud.select_by_index(inliers) # 将属于平面的点提取出来
    planes.append(inlier_cloud)
    planes_model_params.append(plane_model)
    # o3d.visualization.draw_geometries([inlier_cloud]) # 可视化结果
    point_cloud = point_cloud.select_by_index(inliers, invert=True) # 从原始点云中移除属于平面的点

# 2.3 计算各个点云平面的平均距离,并选择平均距离最小者
plane_cnt = len(planes)
print("拟合平面数量为：",plane_cnt)
# print(type(planes[0])) # <class 'open3d.cpu.pybind.geometry.PointCloud'>
average_z = []
point_cnt = []
for i in range(plane_cnt):
    points_tmp = np.asarray(planes[i].points)
    average_z.append(np.mean(points_tmp[:, 2]))
    point_cnt.append(points_tmp.shape[0])
print("各平面中点距离相机的平均距离为",average_z)
print("各平面中点数量",point_cnt)
point_cnt_threshold = 15000 # 用于过滤点数过少的干扰平面
for i in range(plane_cnt):
    if point_cnt[i]<point_cnt_threshold:
        average_z[i] =  float('inf') 
print("各平面中点距离相机的平均距离为(过滤后)",average_z)
min_depth = min(average_z)
min_index = average_z.index(min_depth)
o3d.visualization.draw_geometries([planes[min_index]])

# 3 拟合直线
# 3.1 参数设置
segment = []    # 存储分割结果的容器
min_num = 5000    # 每个分割直线所需的最小点
dist = 0.015      # Ransac分割的距离阈值
iters = 0       # 用于统计迭代次数，非待设置参数
A_save_normalized=[] # 保存斜率 可做方向向量
A_save=[]
B_save=[] # 保存截距
pcd_plane=planes[min_index]
# 3.2 直线拟合
while len(pcd_plane.points) > min_num:
    points = np.asarray(pcd_plane.points)
    line = pyrsc.Line()
    A, B, inliers = line.fit(points, thresh=dist, maxIteration=100) # 拟合一条直线
    line_cloud = pcd_plane.select_by_index(inliers)       # 分割出的直线点云
    r_color = np.random.uniform(0, 1, (1, 3))       # 直线点云随机赋色
    line_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
    pcd_plane = pcd_plane.select_by_index(inliers, invert=True) # 剩余的点云 
    if len(inliers) < min_num: # 如果直线点数小于阈值，则忽略该直线
        break
    segment.append(line_cloud)
    file_name = "RansacFitMutiLine" + str(iters+1) + ".pcd"
    o3d.io.write_point_cloud(file_name, line_cloud)
    iters += 1
    print("A:",A)
    print("B:",B)
    direction_vector = np.asarray(A) # 方向向量
    vector_length = np.linalg.norm(direction_vector) # 计算方向向量的长度（范数
    normalized_direction = direction_vector / vector_length
    A_save_normalized.append(normalized_direction)
    A_save.append(A)
    B_save.append(B)
print("共拟合了",iters,"条有效直线")
# print(A_save_normalized)
# 3.3 直线拟合结果可视化
o3d.visualization.draw_geometries(segment, window_name="Ransac分割多个直线",
                                width=1024, height=768,
                                left=50, top=50,
                                mesh_show_back_face=False)

# 3.4 计算拟合所得直线之间的相似度
lines_vector = np.asarray(A_save_normalized)
# print("lines:",lines_vector)
lines_cnt=len(lines_vector)
similarities = np.zeros((lines_cnt, lines_cnt))
for i in range(lines_cnt):
    for j in range(i, lines_cnt):
        similarity = abs(cosine_similarity(lines_vector[i], lines_vector[j]))
        similarities[i, j] = similarity
        similarities[j, i] = similarity
# 3.5 依据直线方向向量聚类
num_clusters = 2  # 假设将直线分为两类
kmeans = KMeans(n_clusters=num_clusters,n_init= 'auto')
line_classes = kmeans.fit_predict(similarities)
print(line_classes)
print(type(line_classes))
print(line_classes.shape)

# 3.6 找出离群值，消除错误估计结果
threshold=0.997
outliers = []
index_mem =  np.arange(0,line_classes.shape[0])
index_class = []
outliers_index = []
for cluster_id in range(num_clusters):
    index_class.append(index_mem[line_classes == cluster_id])
    cluster_lines = lines_vector[line_classes == cluster_id] # 获取当前类别的直线集合
    for i in range(1,cluster_lines.shape[0],1): # 方向向量存在方向相反情况，将同一类直线的方向向量方向统一，便于求平均
        cosine_compare = cosine_similarity(cluster_lines[0], cluster_lines[i])
        if cosine_compare < 0:
            cluster_lines[i]=-cluster_lines[i]
    mean_vector=(cluster_lines.mean(axis=0)).reshape((1,3))
    centroid_similarity = []
    for i in range(cluster_lines.shape[0]):
        centroid_similarity.append(abs(cosine_similarity(mean_vector, cluster_lines[i])))
    centroid_similarity=np.asarray(centroid_similarity)
    outlier_indices = np.where(centroid_similarity < threshold)[0] # 离群值在当前类中的序号，非所有直线中的序号
    outliers.extend(cluster_lines[outlier_indices]) # 提取离群值对应直线参数
    outliers_index.extend(index_class[cluster_id][outlier_indices]) # 提取离群值标号，在左右直线集合中的序号
print("第一类下标为：",index_class[0])
print("第二类下标为：",index_class[1])
print("离群值数量:", len(outliers)) # 打印离群值数量

for i in range(len(outliers_index)):
    o3d.visualization.draw_geometries([segment[outliers_index[i]]])
for i in range(len(outliers_index)):
    del segment[outliers_index[i]]
    index_class[0] = np.delete(index_class[0], np.where(index_class[0] == outliers_index[i]))
    index_class[1] = np.delete(index_class[1], np.where(index_class[1] == outliers_index[i]))
print("删除离群值后第一类下标为：",index_class[0])
print("删除离群值后第二类下标为：",index_class[1])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
segment.append(mesh_frame)

o3d.visualization.draw_geometries(segment, window_name="Ransac分割多个直线",
                                width=1024, height=768,
                                left=50, top=50,
                                mesh_show_back_face=False)



# o3d.visualization.draw_geometries([planes[min_index]])

np.save("A.npy",A_save)
np.save("B.npy",B_save)
np.save("index_class_0.npy",index_class[0])
np.save("index_class_1.npy",index_class[1])


A_save=np.load("A.npy")
B_save=np.load("B.npy")
index_class_0=index_class[0]
index_class_1=index_class[1]

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
cloud_show.extend(segment)
o3d.visualization.draw_geometries(cloud_show)




# RANSAC单平面分割
'''
import open3d as o3d

# 加载点云
print("->正在加载点云... ")
dataset="point_cloud_00000.pcd"
point_cloud = o3d.io.read_point_cloud(dataset) #点云读取
print(point_cloud)

distance_threshold = 0.01 # 距离阈值，判断是否属于平面模型
ransac_n = 3 # 迭代次数
num_iterations = 100 # 最大迭代次数

# 设置RANSAC参数
plane_model, inliers = o3d.geometry.PointCloud.segment_plane(
    point_cloud,
    distance_threshold=distance_threshold,
    ransac_n=ransac_n,
    num_iterations=num_iterations
)

# 从原始点云中提取属于平面模型的点
inlier_cloud = point_cloud.select_by_index(inliers)

print("->正在可视化点云")
o3d.visualization.draw_geometries([inlier_cloud],window_name = 'cloud_raw')
'''












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