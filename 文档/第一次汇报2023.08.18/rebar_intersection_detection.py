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

# 选择位于顶层平面中的点
def top_plane_select(point_cloud):
    # RANSAC算法参数设置
    distance_threshold = 0.03 # Max distance a point can be from the plane model 0.25 更好 0.3 更容易看出效果
    ransac_n = 3 # Number of initial points to be considered inliers in each iteration.
    num_iterations = 500
    planes = [] # 用于保存平面中的点
    planes_model_params = [] # 用于保存平面参数
    # RANSAC算法依次拟合各个平面
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
        point_cloud = point_cloud.select_by_index(inliers, invert=True) # 从原始点云中移除属于平面的点
    # 计算各个点云平面的平均距离,并选择平均距离最小者
    plane_cnt = len(planes)
    print("->共拟合了",plane_cnt,"个平面")
    # print(type(planes[0])) # <class 'open3d.cpu.pybind.geometry.PointCloud'>
    average_z = []
    point_cnt = []
    for i in range(plane_cnt):
        points_tmp = np.asarray(planes[i].points)
        average_z.append(np.mean(points_tmp[:, 2]))
        point_cnt.append(points_tmp.shape[0])
    print("->各平面中点距离相机的平均距离为",average_z)
    print("->各平面中点数量为",point_cnt)
    point_cnt_threshold = 15000 # 用于过滤点数过少的干扰平面
    for i in range(plane_cnt):
        if point_cnt[i]<point_cnt_threshold:
            average_z[i] =  float('inf') 
    print("->各平面中点距离相机的平均距离为(过滤后)",average_z)
    min_depth = min(average_z)
    min_index = average_z.index(min_depth)
    selected_plane=planes[min_index]
    return planes,planes_model_params,selected_plane

# 直线拟合
def lines_fitting(selected_plane):
    # 参数设置
    segment = []    # 存储分割结果的容器
    min_num = 5000    # 每个分割直线所需的最小点
    dist = 0.015      # Ransac分割的距离阈值 0.013+200次直线迭代最优
    iters = 0       # 用于统计迭代次数，非待设置参数
    A_save_normalized=[] # 保存归一化后的直线方向向量
    A_save=[] # 直线方向向量
    B_save=[] # 保存经过点
    # 直线拟合
    while len(selected_plane.points) > min_num:
        points = np.asarray(selected_plane.points)
        line = pyrsc.Line()
        A, B, inliers = line.fit(points, thresh=dist, maxIteration=200) # 拟合一条直线 A为直线参数方程的方向向量 B为直线经过点
        line_cloud = selected_plane.select_by_index(inliers)       # 分割出的直线点云
        r_color = np.random.uniform(0, 1, (1, 3))       # 直线点云随机赋色
        line_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
        selected_plane = selected_plane.select_by_index(inliers, invert=True) # 剩余的点云 
        if len(inliers) < min_num: # 如果直线点数小于阈值，则忽略该直线
            break
        segment.append(line_cloud)
        # file_name = "RansacFitMutiLine" + str(iters+1) + ".pcd"
        # o3d.io.write_point_cloud(file_name, line_cloud)
        iters += 1
        direction_vector = np.asarray(A) # 方向向量
        vector_length = np.linalg.norm(direction_vector) # 计算方向向量的长度（范数
        normalized_direction = direction_vector / vector_length
        A_save_normalized.append(normalized_direction) # 归一化后的直线方向向量
        A_save.append(A)
        B_save.append(B)
        print("--->已拟合",iters,"条直线")
    print("->共拟合了",iters,"条有效直线")
    # print(A_save_normalized)
    return A_save_normalized,A_save,B_save,segment

# 直线拟合结果显示
def lines_show(index_class,A_save,B_save):
    point_cloud_line_all = [] # 拟合直线点云
    for j in range(2):
        for i in range(index_class[j].shape[0]):
            # 假设参数方程为 P = (b1, b2, b3) + t * (a1, a2, a3)
            P0 = B_save[index_class[j][i]] # 直线经过点
            direction = A_save[index_class[j][i]] # 直线方向向量
            t_values = np.linspace(-0.8, 0.8, num=100)  # 选择一些参数值
            # 计算直线上的点
            line_points = [P0 + t * direction for t in t_values]
            # 创建点云对象
            point_cloud_line = o3d.geometry.PointCloud()
            point_cloud_line.points = o3d.utility.Vector3dVector(line_points)
            line_color = np.random.uniform(0, 1, (1, 3))       # 直线点云随机赋色
            point_cloud_line.paint_uniform_color([line_color[:, 0], line_color[:, 1], line_color[:, 2]])
            point_cloud_line_all.append(point_cloud_line)
    return point_cloud_line_all
    
# 直线离群值筛选
def outlier_filtering(A_save_normalized,segment):
    # 计算拟合所得直线之间的相似度
    lines_vector = np.asarray(A_save_normalized)
    # print("lines:",lines_vector)
    lines_cnt=len(lines_vector)
    similarities = np.zeros((lines_cnt, lines_cnt)) # 用于存储两两直线间的相似度
    for i in range(lines_cnt):
        for j in range(i, lines_cnt):
            similarity = abs(cosine_similarity(lines_vector[i], lines_vector[j])) # 使用预先表示相似度，取模控制取值在[0,1]
            similarities[i, j] = similarity
            similarities[j, i] = similarity
    # 依据直线方向相似度聚类
    num_clusters = 2  # 假设将直线分为两类
    kmeans = KMeans(n_clusters=num_clusters,n_init= 'auto')
    line_classes = kmeans.fit_predict(similarities)
    print("->直线分类结果为：",line_classes)

    # 找出离群值，消除错误估计结果
    threshold=0.997 # 相似度阈值，小于该阈值认为是离群值
    outliers = [] # 存储离群直线方向向量
    index_mem =  np.arange(0,line_classes.shape[0]) # 保存直线标号
    index_class = [] # 用于存储各类直线的下标
    outliers_index = [] # 离群直线标号
    for cluster_id in range(num_clusters):
        index_class.append(index_mem[line_classes == cluster_id])
        cluster_lines = lines_vector[line_classes == cluster_id] # 获取当前类别的直线集合
        for i in range(1,cluster_lines.shape[0],1): # 方向向量存在方向相反情况，将同一类直线的方向向量方向统一，便于求平均
            cosine_compare = cosine_similarity(cluster_lines[0], cluster_lines[i])
            if cosine_compare < 0:
                cluster_lines[i]=-cluster_lines[i]
        mean_vector=(cluster_lines.mean(axis=0)).reshape((1,3)) # 求某一类直线的均值方向向量
        centroid_similarity = [] # 用于存储各直线与均值方向向量之间的相似度
        for i in range(cluster_lines.shape[0]): # 计算直线与均值方向向量之间的相似度
            centroid_similarity.append(abs(cosine_similarity(mean_vector, cluster_lines[i])))
        centroid_similarity=np.asarray(centroid_similarity)
        outlier_indices = np.where(centroid_similarity < threshold)[0] # 离群值在当前类中的序号，非所有直线中的序号
        outliers.extend(cluster_lines[outlier_indices]) # 提取离群直线方向向量
        outliers_index.extend(index_class[cluster_id][outlier_indices]) # 提取离群值标号，在左右直线集合中的序号
    print("->第一类下标为：",index_class[0])
    print("->第二类下标为：",index_class[1])
    print("->离群值数量:", len(outliers)) # 打印离群值数量

    for i in range(len(outliers_index)):
        o3d.visualization.draw_geometries([segment[outliers_index[i]]], window_name="离群直线包含的点")
    for i in range(len(outliers_index)): # 从各类直线标号集合中删除离群直线的标号 删除离群直线对应点云
        index_class[0] = np.delete(index_class[0], np.where(index_class[0] == outliers_index[i]))
        index_class[1] = np.delete(index_class[1], np.where(index_class[1] == outliers_index[i]))
    for index in sorted(outliers_index, reverse=True):
        del segment[index]
    print("->删除离群值后第一类下标为：",index_class[0])
    print("->删除离群值后第二类下标为：",index_class[1])
    return index_class,segment

# 直线交叉点检测
def crossing_compute(index_class,A_save,B_save):
    cross_point = []
    # x = a_1*t+b_1
    # y = a_2*t+b_2
    # z = a_3*t+b_3
    t_1, t_2 = symbols('t_1 t_2')
    cross_point_temp=[]
    cross_cnt=0
    for m in range(index_class[0].shape[0]):
        for n in range(index_class[1].shape[0]):
            result= solve([A_save[index_class[0][m]][0]*t_1+B_save[index_class[0][m]][0]-(A_save[index_class[1][n]][0]*t_2+B_save[index_class[1][n]][0]),A_save[index_class[0][m]][1]*t_1+B_save[index_class[0][m]][1]-(A_save[index_class[1][n]][1]*t_2+B_save[index_class[1][n]][1])],[t_1, t_2])
            # print(result)
            cross_point_x=A_save[index_class[0][m]][0]*result[t_1]+B_save[index_class[0][m]][0]
            cross_point_y=A_save[index_class[0][m]][1]*result[t_1]+B_save[index_class[0][m]][1]
            cross_point_z_1=A_save[index_class[0][m]][2]*result[t_1]+B_save[index_class[0][m]][2]
            cross_point_z_2=A_save[index_class[1][n]][2]*result[t_2]+B_save[index_class[1][n]][2]
            cross_point_z_mean=(cross_point_z_2+cross_point_z_1)/2
            cross_point_temp.append(np.array([cross_point_x,cross_point_y,cross_point_z_mean]))
            cross_cnt=cross_cnt+1
    print("->共",cross_cnt,"个交点")
    # print(cross_point_temp)
    cross_point_set = np.array(cross_point_temp)
    # print(cross_point_set)
    # cross_point_temp = np.random.rand(100, 3)
    # print('type(cross_point_temp):',type(cross_point_temp))
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
    return grasp_point_mark

# 1.加载点云
print("-------------step 1:加载点云---------------")
print("->正在加载点云... ")
dataset="point_cloud_00000.pcd"
point_cloud_pcd = o3d.io.read_point_cloud(dataset) #点云读取
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3) # 显示坐标系
cloud_point_show=[]
cloud_point_show.append(point_cloud_pcd)
cloud_point_show.append(mesh_frame)
o3d.visualization.draw_geometries(cloud_point_show, window_name="原始点云数据")
print("->点云加载成功:",point_cloud_pcd)

# 2.最上层钢筋点云分离
print("-------step 2:最上层钢筋点云分离-----------")
planes_out,planes_model_params_out,selected_plane_out=top_plane_select(point_cloud=point_cloud_pcd) # Visualization_on 是否显示平面分割中间结果
o3d.visualization.draw_geometries([selected_plane_out], window_name="选择的顶层平面点")
for i in range(len(planes_out)):
    o3d.visualization.draw_geometries([planes_out[i]], window_name="平面"+str(i))

# 3 拟合直线
print("-------------step 3:拟合直线---------------")
A_save_normalized_out,A_save_out,B_save_out,segment_out = lines_fitting(selected_plane_out)
# np.save("A.npy",A_save)
# np.save("B.npy",B_save)
# 直线拟合结果可视化
o3d.visualization.draw_geometries(segment_out, window_name="直线分割结果")

# 4 直线分类并删除离群值
print("--------step 4:直线分类并删除离群值---------")
index_class_out,segment_out=outlier_filtering(A_save_normalized_out,segment_out)
# np.save("index_class[0].npy",index_class[0])
# np.save("index_class[1].npy",index_class[1])
cloud_point_show=[]
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3) # 显示坐标系
cloud_point_show.append(mesh_frame)
cloud_point_show.extend(segment_out)
o3d.visualization.draw_geometries(cloud_point_show, window_name="直线分割结果（删除离群直线后）")

# 5 直线拟合结果绘制
# A_save=np.load("A.npy")
# B_save=np.load("B.npy")
point_cloud_line_all=lines_show(index_class_out,A_save_out,B_save_out)
o3d.visualization.draw_geometries(point_cloud_line_all)

# 6 直线交叉点求解
grasp_point_mark_out=crossing_compute(index_class_out,A_save_out,B_save_out)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
cloud_show=[]
cloud_show.extend(point_cloud_line_all)
cloud_show.extend(grasp_point_mark_out)
cloud_show.append(mesh_frame)
o3d.visualization.draw_geometries(cloud_show) 
cloud_show.extend(segment_out)
o3d.visualization.draw_geometries(cloud_show) 