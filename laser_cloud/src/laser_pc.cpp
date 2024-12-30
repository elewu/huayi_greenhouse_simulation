#include <iostream>

#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/Imu.h"
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <laser_geometry/laser_geometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <visualization_msgs/Marker.h>

#include <algorithm>
#include <mutex>//互斥锁，单次一个函数处理雷达消息
#include <thread>
#include <condition_variable>
#include <vector>

//判断单个点是否有效
bool isValidPoint(const pcl::PointXYZI &point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}
//求解点云y坐标平均值
double pointcloud_ymean(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
  double y_average = 0;
  double y_sum = 0.0;
  int point_count = 0;
  for (const auto &point : *cloud) {
    y_sum += point.y;
    ++point_count;
  }
  return y_average;
}
//求解cloud中x的最小值
float pointcloud_xmin(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
  float x_min = 20;
  for (const auto &point : *cloud) {
    float temp = point.x;
    if (temp<x_min){
      x_min=temp;
    }}
  return x_min;
}

float pointcloud_xmax(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
  float x_max = -20;
  for (const auto &point : *cloud) {
    float temp = point.x;
    if (temp>x_max){
      x_max=temp;
    }}
  return x_max;
}

float calculateY(pcl::ModelCoefficients::Ptr coefficients, float x, float z) {
  float y = (-coefficients->values[0] * x - coefficients->values[2] * z -
             coefficients->values[3]) /
            coefficients->values[1];
  return y;
}

class laser_pc {
private:
  ros::NodeHandle nh;
  ros::Subscriber laser_scan1, laser_scan2, odometry_sub;
  ros::Publisher pointcloud_pub1, pointcloud_pub2, Markerleft_pub,
      Markerright_pub, Markermid_pub,velcmd_pub; //发布laser_scan转换后的pointcloud
  sensor_msgs::LaserScan laser_f, laser_b;
  pcl::PointCloud<pcl::PointXYZI> cloud_merge_l; //保存左侧点云
  pcl::PointCloud<pcl::PointXYZI> cloud_merge_r; //保存右侧点云
  float odom[3]={0,0,0};//记录一下机器人走过的路程
  std::vector<float> error_p;
  int speed_flag=0;
  int process_flag=0;
  int cmd_flag=0;
  std::mutex mutex;


  //该处定义PID的相关参数
  double pids[3]={0,0,0};
  bool ifget1=nh.getParam("kp_s", pids[0]);
  bool ifget2=nh.getParam("ki_s", pids[1]);
  bool ifget3=nh.getParam("kd_s", pids[2]);
  double pidb[3]={0,0,0};
  bool ifget4=nh.getParam("kp_b", pidb[0]);
  bool ifget5=nh.getParam("ki_b", pidb[1]);
  bool ifget6=nh.getParam("kd_b", pidb[2]);
  double kp_ = 0;
  double ki_ = 0;
  double kd_ = 0;
  double integral_ = 0;
  double last_error_ = 0;

public:
  laser_pc() {
    laser_scan1 = nh.subscribe<sensor_msgs::LaserScan>(
        "laser1_scan", 1000, &laser_pc::callback1, this);
    laser_scan2 = nh.subscribe<sensor_msgs::LaserScan>(
        "laser2_scan", 1000, &laser_pc::callback2, this);
    odometry_sub = nh.subscribe<nav_msgs::Odometry>(
        "odom", 1000, &laser_pc::callback0, this);   
    pointcloud_pub1 =
        nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("pointcloud_1", 1000);
    pointcloud_pub2 =
        nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("pointcloud_2", 1000);
    Markerleft_pub =
        nh.advertise<visualization_msgs::Marker>("marker_left", 1000);
    Markerright_pub =
        nh.advertise<visualization_msgs::Marker>("marker_right", 1000);
    Markermid_pub =
        nh.advertise<visualization_msgs::Marker>("marker_mid", 1000);
    velcmd_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1000);
    cloud_merge_l.header.frame_id = "base_link";
    cloud_merge_r.header.frame_id = "base_link";
  }

  void callback0(const nav_msgs::Odometry::ConstPtr &odom);
  void callback1(const sensor_msgs::LaserScan::ConstPtr &laser1);
  void callback2(const sensor_msgs::LaserScan::ConstPtr &laser2);

  //线程处理函数，分别作点云处理和运动控制
  void PoinCloudProcessing();
  void VelocityCommand();
  
  void
  LaserScanToPointCloud(sensor_msgs::LaserScan _laser_scan,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &_pointcloud);
  void
  ransacFitting(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input_cloud,
                Eigen::VectorXf *coefficient, std::vector<int> *inliers);
  void PointCloud_preprocess(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_filtered);
  void visualizeLine(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                     float *coeffients, std::vector<int> &inliers,
                     const std::string &frame_id,
                     const ros::Publisher &marker_pub,
                     float* x_min_max);
  void clusterWalls(const pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_cluster1,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_cluster2);
  pcl::PointCloud<pcl::PointXYZI>::Ptr
  filterPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud);
  void MidlineMarkerVisualize(pcl::ModelCoefficients::Ptr &coefficients_mid,
                              ros::Publisher &publisher);
  double calculateOutput(double setpoint, double current_value, float speed);
  void velocitysetting(geometry_msgs::Twist::Ptr velocity, double linear_x, double angle_z);
};
// laserscan转pointcloud<pcl::XYZI>
void laser_pc::LaserScanToPointCloud(
    sensor_msgs::LaserScan _laser_scan,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &_pointcloud) {
  _pointcloud->clear();
  pcl::PointXYZI newPoint;
  newPoint.z = 0.0;
  double newPointAngle;

  int beamNum = _laser_scan.ranges.size();
  for (int i = 0; i < beamNum; i++) {
    newPointAngle = _laser_scan.angle_min + _laser_scan.angle_increment * i;
    newPoint.x = _laser_scan.ranges[i] * cos(newPointAngle);
    newPoint.y = _laser_scan.ranges[i] * sin(newPointAngle);
    newPoint.intensity = _laser_scan.intensities[i];
    _pointcloud->push_back(newPoint);
  }
}
//点云预处理：下采样和滤波
void laser_pc::PointCloud_preprocess(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_filtered) {
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(0.01f, 0.01f, 0.01f);
  sor.filter(*cloud_filtered);
}

// RANSAC拟合
void laser_pc::ransacFitting(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input_cloud,
    Eigen::VectorXf *coefficient, std::vector<int> *inliers) {
  pcl::SampleConsensusModelLine<pcl::PointXYZI>::Ptr line(
      new pcl::SampleConsensusModelLine<pcl::PointXYZI>(input_cloud));
  pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(line);
  ransac.setDistanceThreshold(0.01);
  ransac.setMaxIterations(1000);
  ransac.computeModel();

  ransac.getModelCoefficients(*coefficient);
  ransac.getInliers(*inliers);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_line(
      new pcl::PointCloud<pcl::PointXYZI>); //用来保存内点
  pcl::copyPointCloud<pcl::PointXYZI>(*input_cloud, *inliers, *cloud_line);
}

// X方向范围滤波
pcl::PointCloud<pcl::PointXYZI>::Ptr laser_pc::filterPointCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PassThrough<pcl::PointXYZI> pass;
  pass.setInputCloud(input_cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(-2.5, 2.5);
  pass.setFilterLimitsNegative(false);
  pass.filter(*cloud_filtered);

  return cloud_filtered;
}
//聚类
void laser_pc::clusterWalls(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_cluster1,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_cluster2) {
  // 过滤无效点
  std::vector<int> indices_to_remove;
  for (size_t i = 0; i < input_cloud->points.size(); ++i) {
    if (!std::isfinite(input_cloud->points[i].x) ||
        !std::isfinite(input_cloud->points[i].y) ||
        !std::isfinite(input_cloud->points[i].z)) {
      indices_to_remove.push_back(i);
      // ROS_INFO("%d",i);
    }
  }
  for (int i = indices_to_remove.size() - 1; i >= 0; --i) {
    input_cloud->erase(input_cloud->begin() + indices_to_remove[i]);
  }
  // pcl::removeNaNFromPointCloud(*input_cloud, indices_to_remove);

  // 创建KdTree对象，用于加速聚类过程中的近邻搜索
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZI>);
  tree->setInputCloud(input_cloud);

  // 创建欧氏距离聚类对象
  pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
  ec.setClusterTolerance(0.1);   // 设置簇的最大距离阈值
  ec.setMinClusterSize(20);      // 设置最小簇大小
  ec.setMaxClusterSize(25000);   // 设置最大簇大小
  ec.setSearchMethod(tree);      // 设置搜索方法为KdTree
  ec.setInputCloud(input_cloud); // 设置输入点云

  // 执行聚类
  std::vector<pcl::PointIndices> cluster_indices;
  ec.extract(cluster_indices);

  // 处理聚类结果
  if (cluster_indices.size() >= 2) {
    // 保存第一个簇
    cloud_cluster1.reset(new pcl::PointCloud<pcl::PointXYZI>);
    for (std::vector<int>::const_iterator pit =
             cluster_indices[0].indices.begin();
         pit != cluster_indices[0].indices.end(); ++pit) {
      cloud_cluster1->points.push_back(input_cloud->points[*pit]);
    }
    // 保存第二个簇
    cloud_cluster2.reset(new pcl::PointCloud<pcl::PointXYZI>);
    for (std::vector<int>::const_iterator pit =
             cluster_indices[1].indices.begin();
         pit != cluster_indices[1].indices.end(); ++pit) {
      cloud_cluster2->points.push_back(input_cloud->points[*pit]);
    }
  } else {
    std::cout << "Clustering failed, fewer than 2 clusters found." << std::endl;
    // 聚类失败，可以在这里添加适当的处理逻辑
  }
  
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster3(
      new pcl::PointCloud<pcl::PointXYZI>);
  double y1 = pointcloud_ymean(cloud_cluster1);
  double y2 = pointcloud_ymean(cloud_cluster2);
  if (y1 < y2) {
    cloud_cluster3 = cloud_cluster1;
    cloud_cluster1 = cloud_cluster2;
    cloud_cluster2 = cloud_cluster3;
  } //保证cloud_cluster1的y>cloud_cluster2，1为左侧点云
  cloud_cluster1->header.frame_id = "base_link";
  cloud_cluster2->header.frame_id = "base_link";
}
//墙壁边线可视化，查看边墙的前后点
void laser_pc::visualizeLine(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                             float *coeffients, std::vector<int> &inliers,
                             const std::string &frame_id,
                             const ros::Publisher &marker_pub, float* x_min_max) {
  visualization_msgs::Marker line;
  line.header.frame_id = frame_id;
  line.header.stamp = ros::Time::now();
  line.ns = "line";
  line.action = visualization_msgs::Marker::ADD;
  line.pose.orientation.w = 1.0;
  line.id = 0;
  line.type = visualization_msgs::Marker::LINE_STRIP;
  line.scale.x = 0.015;
  line.color.g = 1.0;
  line.color.a = 1.0;

  int minValue = *std::min_element(inliers.begin(), inliers.end());
  int maxValue = *std::max_element(inliers.begin(), inliers.end());

  pcl::PointXYZI point1 = cloud->points[minValue];
  geometry_msgs::Point p1;
  p1.x = point1.x;
  p1.y = point1.y;
  p1.z = point1.z;
  line.points.push_back(p1);

  pcl::PointXYZI point2 = cloud->points[maxValue];
  geometry_msgs::Point p2;
  p2.x = point2.x;
  p2.y = point2.y;
  p2.z = point2.z;
  line.points.push_back(p2);

  x_min_max[0]=p1.x;
  x_min_max[1]=p2.x;

  float k = (p2.y - p1.y) / (p2.x - p1.x);
  float b = p1.y - k * p1.x;

  coeffients[0] = k;
  coeffients[1] = b;

  marker_pub.publish(line);
}
//中线可视化
void laser_pc::MidlineMarkerVisualize(
    pcl::ModelCoefficients::Ptr &coefficients_mid, ros::Publisher &publisher) {
  visualization_msgs::Marker line;
  line.header.frame_id = "base_link";
  line.header.stamp = ros::Time::now();
  line.ns = "line";
  line.action = visualization_msgs::Marker::ADD;
  line.pose.orientation.w = 1.0;
  line.id = 0;
  line.type = visualization_msgs::Marker::LINE_STRIP;
  line.scale.x = 0.05;
  line.color.r = 1.0;
  line.color.a = 1.0;
  float x = 0.5;
  for (int i = 0; i < 12; ++i) {
    float y = calculateY(coefficients_mid, x, 0.05);
    geometry_msgs::Point p;
    p.x = x;
    p.y = y;
    p.z = 0.05;
    line.points.push_back(p);
    x = x + 0.01;
  }
  publisher.publish(line);
}

float distance(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
               std::vector<int> &inliers) {
  float mindistance = 20;
  for (int i : inliers) {
    pcl::PointXYZI point = cloud->points[i];
    float distance = sqrt(point.x * point.x + point.y * point.y);
    if (distance < mindistance) {
      mindistance = distance;
    }
  }
  return mindistance;
}
// PID输出角速度
double laser_pc::calculateOutput(double setpoint, double current_value, float speed) {
  if (speed=1.0){
    kp_ = pids[0];
    ki_ = pids[1];
    kd_ = pids[2];
  }
  else{
  kp_ = pidb[0];
  ki_ = pidb[1];
  kd_ = pidb[2];
  }
  double error = setpoint - current_value;
  integral_ += error;
  double derivative = (error - last_error_);
  double output = kp_ * error + ki_ * integral_ + kd_ * derivative;
  last_error_ = error;
  return output;
}

void laser_pc::velocitysetting(geometry_msgs::Twist::Ptr velocity, double linear_x, double angle_z)
{
  velocity->linear.x=linear_x;
  velocity->linear.y=0;
  velocity->linear.z=0;
  velocity->angular.x=0;
  velocity->angular.y=0;
  velocity->angular.z=angle_z;
}

//保存odom消息
void laser_pc::callback0(const nav_msgs::Odometry::ConstPtr &odom_origin){
  std::unique_lock<std::mutex> lock(mutex);
  odom[0]=odom_origin->pose.pose.position.x;
  odom[1]=odom_origin->pose.pose.position.y;
  odom[2]=odom_origin->pose.pose.position.z;
}

//保存前侧雷达
void laser_pc::callback1(const sensor_msgs::LaserScan::ConstPtr &laser1) {
  std::unique_lock<std::mutex> lock(mutex);
  laser_f=*laser1;
  process_flag++;
}

//保存后侧雷达
void laser_pc::callback2(const sensor_msgs::LaserScan::ConstPtr &laser2) {
  std::unique_lock<std::mutex> lock(mutex);
  laser_b=*laser2;
  process_flag++;
}

void laser_pc::PoinCloudProcessing(){
  std::unique_lock<std::mutex> lock(mutex);
  ros::Rate r(100);
  if(process_flag==2){
    process_flag=0;
    //ROS_INFO("PROCESS_FLAG:%d", process_flag);
    cmd_flag=1;
    pcl::PointCloud<pcl::PointXYZI>::Ptr _pointcloud_f(new pcl::PointCloud<pcl::PointXYZI>);
    LaserScanToPointCloud(laser_f, _pointcloud_f); 
    pcl_conversions::toPCL(laser_f.header, _pointcloud_f->header);
    cloud_merge_l.clear();
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pointcloud_f(
        new pcl::PointCloud<pcl::PointXYZI>);
    transformed_pointcloud_f->header.frame_id = "base_link";

    Eigen::Affine3f transform_f = Eigen::Affine3f::Identity();
    transform_f.translation() << 0.16902, 0.0, 0.054;
    transform_f.rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*_pointcloud_f, *transformed_pointcloud_f,
                            transform_f);
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pointcloud_fl(
        new pcl::PointCloud<pcl::PointXYZI>); //前方左侧点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pointcloud_fr(
        new pcl::PointCloud<pcl::PointXYZI>); //前方右侧点云
    transformed_pointcloud_f = filterPointCloud(transformed_pointcloud_f);

    clusterWalls(transformed_pointcloud_f, transformed_pointcloud_fl,
                transformed_pointcloud_fr);
    cloud_merge_l = *transformed_pointcloud_fl;
    cloud_merge_r = *transformed_pointcloud_fr;

    pcl::PointCloud<pcl::PointXYZI>::Ptr _pointcloud_b(new pcl::PointCloud<pcl::PointXYZI>);
    LaserScanToPointCloud(laser_b, _pointcloud_b);
    pcl_conversions::toPCL(laser_b.header, _pointcloud_b->header);
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pointcloud_b(new pcl::PointCloud<pcl::PointXYZI>);
    transformed_pointcloud_b->header.frame_id = "base_link";

    Eigen::Affine3f transform_b = Eigen::Affine3f::Identity();
    transform_b.translation() << -0.33546, 0.0, 0.0471;
    transform_b.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()));
    pcl::transformPointCloud(*_pointcloud_b, *transformed_pointcloud_b, transform_b);

    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pointcloud_bl(
        new pcl::PointCloud<pcl::PointXYZI>); //前方左侧点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pointcloud_br(
        new pcl::PointCloud<pcl::PointXYZI>); //前方右侧点云
    transformed_pointcloud_b = filterPointCloud(
        transformed_pointcloud_b); //后方点云需要滤除远端无用的点云
    clusterWalls(transformed_pointcloud_b, transformed_pointcloud_br,
                transformed_pointcloud_bl);
    cloud_merge_l = cloud_merge_l + *transformed_pointcloud_bl;
    cloud_merge_r = cloud_merge_r + *transformed_pointcloud_br;
    pointcloud_pub1.publish(cloud_merge_l);
    pointcloud_pub2.publish(cloud_merge_r);
    //到这里获取到了左侧完整点云cloud_merge_l和cloud_merge_r
  }
  r.sleep();
}

void laser_pc::VelocityCommand(){
  ros::Rate rate(100); // 雷达数据处理频率
  if (cmd_flag==1) {
    std::lock_guard<std::mutex> lock(mutex);
    cmd_flag=0;
    Eigen::VectorXf coefficients_left; //左侧点云拟合所需的参数
    std::vector<int> inliers_left;
    Eigen::VectorXf coefficients_right; //右侧点云拟合所需的参数
    std::vector<int> inliers_right;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_left(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_right(
        new pcl::PointCloud<pcl::PointXYZI>);
    *cloud_left = cloud_merge_l;
    *cloud_right = cloud_merge_r;
    //对左右两块点云进行直线拟合
    ransacFitting(cloud_left, &coefficients_left, &inliers_left);
    ransacFitting(cloud_right, &coefficients_right, &inliers_right);

    //拟合结果可视化
    float coeff_l[2] = {0, 0};
    float coeff_r[2] = {0, 0};
    //保存左右边墙x方向的最大值和最小值
    float wallx_left[2]={0,0};
    float wallx_right[2]={0,0};
    wallx_left[0]=pointcloud_xmin(cloud_left); wallx_left[1]=pointcloud_xmax(cloud_left);
    wallx_right[0]=pointcloud_xmin(cloud_right); wallx_right[1]=pointcloud_xmax(cloud_right);
    float line_min_max_left[2]={0,0};
    float line_min_max_right[2]={0,0};
    visualizeLine(cloud_left, coeff_l, inliers_left, "base_link", Markerleft_pub, line_min_max_left);
    visualizeLine(cloud_right, coeff_r, inliers_right, "base_link",
                  Markerright_pub, line_min_max_right);

    float coeff_m[2];
    coeff_m[0] = (coeff_l[0] + coeff_r[0]) / 2;
    coeff_m[1] = (coeff_l[1] + coeff_r[1]) / 2;
    visualization_msgs::Marker line;
    line.header.frame_id = "base_link";
    line.header.stamp = ros::Time::now();
    line.ns = "line";
    line.action = visualization_msgs::Marker::ADD;
    line.pose.orientation.w = 1.0;
    line.id = 0;
    line.type = visualization_msgs::Marker::LINE_STRIP;
    line.scale.x = 0.015;
    line.color.g = 1.0;
    line.color.a = 1.0;

    geometry_msgs::Point point1, point2;
    point1.x = -1;
    point1.y = coeff_m[0] * point1.x + coeff_m[1];
    point1.z = 0.05;
    point2.x = 2;
    point2.y = coeff_m[0] * point2.x + coeff_m[1];
    point2.z = 0.05;
    line.points.push_back(point2);
    line.points.push_back(point1);
    Markermid_pub.publish(line);

    float target_x = 1.5;
    //用来判断进出墙壁通道，靠前后雷达的点云x坐标信息，这里没有用到车体下方雷达  
    if(wallx_left[0]<-0.6 && wallx_right[0]<-0.6){
      speed_flag=1;
    }
    if(wallx_left[1]<0.2 && wallx_right[1]<0.2 && odom[0]>12.8){
      
      speed_flag=2;
    }
    if(speed_flag==0){
      geometry_msgs::Twist::Ptr velocity(new geometry_msgs::Twist);
      velocitysetting(velocity, 0.4, calculateOutput(target_x * coeff_m[0] + coeff_m[1], 0, 0.4));
      velcmd_pub.publish(velocity);}

    if(speed_flag==1){
      geometry_msgs::Twist::Ptr velocity(new geometry_msgs::Twist);
      velocitysetting(velocity, 1.0, calculateOutput(target_x * coeff_m[0] + coeff_m[1], 0, 1.0));
      velcmd_pub.publish(velocity);}  

    if(speed_flag==2){
      geometry_msgs::Twist::Ptr velocity(new geometry_msgs::Twist);
      velocitysetting(velocity, 0.0, 0.0);
      velcmd_pub.publish(velocity);}      
      
      rate.sleep();
   }  
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "laser_pc");

  laser_pc laser_cloud;
  while(ros::ok())
  {
    ros::spinOnce();//处理回调函数
    //线程单独各自跑  
    std::thread processThread(&laser_pc::PoinCloudProcessing, &laser_cloud);
    std::thread velcmdThread(&laser_pc::VelocityCommand, &laser_cloud);
    processThread.join();
    velcmdThread.join();
    //释放线程
  }
  
  return 0;
}