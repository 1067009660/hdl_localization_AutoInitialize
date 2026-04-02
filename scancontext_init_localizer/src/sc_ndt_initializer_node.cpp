#include "scancontext_init_localizer/EstimateInitialPose.h"
#include "scancontext_init_localizer/scan_context.hpp"

#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/registration.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pclomp/ndt_omp.h>
#include <ros/ros.h>
#include <ros/topic.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/LinearMath/Quaternion.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace scl = scancontext_init_localizer;

class SCNDTInitializer {
public:
  SCNDTInitializer() : nh_(), pnh_("~") {
    pnh_.param<std::string>("db_dir", db_dir_, "/tmp/scancontext_db");
    pnh_.param<std::string>("lidar_topic", lidar_topic_, "/velodyne_points");
    pnh_.param<std::string>("map_frame", map_frame_, "map");
    pnh_.param<std::string>("initialpose_topic", initialpose_topic_, "/initialpose");
    pnh_.param<double>("service_wait_cloud_sec", wait_cloud_sec_, 1.0);
    pnh_.param<int>("num_candidates", num_candidates_, 10);
    pnh_.param<double>("sc_distance_threshold", sc_dist_threshold_, 0.25);

    pnh_.param<double>("ndt_resolution", ndt_resolution_, 1.0);
    pnh_.param<double>("ndt_step_size", ndt_step_size_, 0.1);
    pnh_.param<double>("ndt_trans_eps", ndt_trans_eps_, 0.01);
    pnh_.param<int>("ndt_max_iter", ndt_max_iter_, 40);
    pnh_.param<int>("ndt_num_threads", ndt_num_threads_, 4);
    pnh_.param<int>("ndt_neighborhood", ndt_neighborhood_, 2);

    scl::ScanContext::Params sc_params;
    pnh_.param<int>("num_rings", sc_params.num_rings, 20);
    pnh_.param<int>("num_sectors", sc_params.num_sectors, 60);
    pnh_.param<double>("max_radius", sc_params.max_radius, 80.0);
    pnh_.param<double>("lidar_height", sc_params.lidar_height, 2.0);
    sc_.reset(new scl::ScanContext(sc_params));

    std::string error;
    if (!scl::loadDatabase(db_dir_, sc_params.num_rings, sc_params.num_sectors, &entries_, &error)) {
      ROS_FATAL_STREAM("Failed to load SC database: " << error << " db_dir=" << db_dir_);
      ros::shutdown();
      return;
    }

    initialpose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(initialpose_topic_, 1, true);
    service_ = nh_.advertiseService("scancontext_initialize", &SCNDTInitializer::serviceCallback, this);

    ROS_INFO_STREAM("sc_ndt_initializer ready: entries=" << entries_.size() << " service=/scancontext_initialize");
  }

private:
  bool serviceCallback(scancontext_init_localizer::EstimateInitialPose::Request& req,
                       scancontext_init_localizer::EstimateInitialPose::Response& res) {
    (void)req;

    const auto cloud_msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(
        lidar_topic_, nh_, ros::Duration(wait_cloud_sec_));
    if (!cloud_msg) {
      res.success = false;
      res.message = "timeout waiting lidar topic";
      return true;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*cloud_msg, *source_cloud);
    if (source_cloud->empty()) {
      res.success = false;
      res.message = "empty lidar cloud";
      return true;
    }

    const auto query_desc = sc_->makeDescriptor(*source_cloud);
    const auto query_key = sc_->makeRingKey(query_desc);

    std::vector<std::pair<float, int>> key_dists;
    key_dists.reserve(entries_.size());
    for (size_t i = 0; i < entries_.size(); ++i) {
      key_dists.emplace_back(sc_->ringKeyDistance(query_key, entries_[i].ring_key), static_cast<int>(i));
    }
    std::sort(key_dists.begin(), key_dists.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    const int eval_num = std::min<int>(num_candidates_, static_cast<int>(key_dists.size()));
    float best_dist = std::numeric_limits<float>::max();
    int best_idx = -1;
    int best_shift = 0;

    for (int i = 0; i < eval_num; ++i) {
      const int idx = key_dists[i].second;
      const auto dist_shift = sc_->distanceWithYaw(query_desc, entries_[idx].descriptor);
      if (dist_shift.first < best_dist) {
        best_dist = dist_shift.first;
        best_idx = idx;
        best_shift = dist_shift.second;
      }
    }

    if (best_idx < 0 || best_dist > sc_dist_threshold_) {
      res.success = false;
      res.sc_distance = best_dist;
      res.matched_index = best_idx >= 0 ? entries_[best_idx].index : -1;
      res.message = "SC matching failed or over threshold";
      return true;
    }

    const auto& matched = entries_[best_idx];
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZI>());

    std::string target_path = matched.pcd_path;
    if (!boost::filesystem::path(target_path).is_absolute()) {
      target_path = (boost::filesystem::path(db_dir_) / target_path).string();
    }
    if (pcl::io::loadPCDFile(target_path, *target_cloud) != 0 || target_cloud->empty()) {
      res.success = false;
      res.message = "failed to load matched target pcd";
      return true;
    }

    const double yaw_delta = sc_->sectorAngleRad() * static_cast<double>(best_shift);
    tf2::Quaternion q_guess;
    q_guess.setRPY(0.0, 0.0, yaw_delta);
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf qf(static_cast<float>(q_guess.w()),
                          static_cast<float>(q_guess.x()),
                          static_cast<float>(q_guess.y()),
                          static_cast<float>(q_guess.z()));
    init_guess.block<3, 3>(0, 0) = qf.toRotationMatrix();

    pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    ndt.setResolution(ndt_resolution_);
    ndt.setStepSize(ndt_step_size_);
    ndt.setTransformationEpsilon(ndt_trans_eps_);
    ndt.setMaximumIterations(ndt_max_iter_);
    ndt.setNumThreads(ndt_num_threads_);

    if (ndt_neighborhood_ == 0) {
      ndt.setNeighborhoodSearchMethod(pclomp::KDTREE);
    } else if (ndt_neighborhood_ == 1) {
      ndt.setNeighborhoodSearchMethod(pclomp::DIRECT26);
    } else if (ndt_neighborhood_ == 3) {
      ndt.setNeighborhoodSearchMethod(pclomp::DIRECT1);
    } else {
      ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    }

    ndt.setInputTarget(target_cloud);
    ndt.setInputSource(source_cloud);

    pcl::PointCloud<pcl::PointXYZI> aligned;
    ndt.align(aligned, init_guess);

    if (!ndt.hasConverged()) {
      res.success = false;
      res.sc_distance = best_dist;
      res.matched_index = matched.index;
      res.message = "NDT did not converge";
      return true;
    }

    const Eigen::Matrix4f tf_historical_current = ndt.getFinalTransformation();
    const Eigen::Matrix4f tf_map_historical = composePoseMatrix(matched.position, matched.orientation);
    const Eigen::Matrix4f tf_map_current = tf_map_historical * tf_historical_current;

    Eigen::Quaternionf q(tf_map_current.block<3, 3>(0, 0));
    q.normalize();

    geometry_msgs::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = ros::Time::now();
    pose_msg.header.frame_id = map_frame_;
    pose_msg.pose.pose.position.x = tf_map_current(0, 3);
    pose_msg.pose.pose.position.y = tf_map_current(1, 3);
    pose_msg.pose.pose.position.z = tf_map_current(2, 3);
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    for (double& c : pose_msg.pose.covariance) {
      c = 0.0;
    }
    pose_msg.pose.covariance[0] = 0.25;
    pose_msg.pose.covariance[7] = 0.25;
    pose_msg.pose.covariance[35] = 0.2;

    initialpose_pub_.publish(pose_msg);

    res.success = true;
    res.message = "success";
    res.initial_pose = pose_msg;
    res.sc_distance = best_dist;
    res.matched_index = matched.index;

    ROS_INFO_STREAM("Init pose published. idx=" << matched.index << " sc_dist=" << best_dist
                    << " ndt_fitness=" << ndt.getFitnessScore());
    return true;
  }

private:
  static Eigen::Matrix4f composePoseMatrix(const Eigen::Vector3d& t, const Eigen::Quaterniond& q) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf qf(static_cast<float>(q.w()),
                          static_cast<float>(q.x()),
                          static_cast<float>(q.y()),
                          static_cast<float>(q.z()));
    qf.normalize();
    m.block<3, 3>(0, 0) = qf.toRotationMatrix();
    m(0, 3) = static_cast<float>(t.x());
    m(1, 3) = static_cast<float>(t.y());
    m(2, 3) = static_cast<float>(t.z());
    return m;
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Publisher initialpose_pub_;
  ros::ServiceServer service_;

  std::unique_ptr<scl::ScanContext> sc_;
  std::vector<scl::DatabaseEntry> entries_;

  std::string db_dir_;
  std::string lidar_topic_;
  std::string map_frame_;
  std::string initialpose_topic_;
  double wait_cloud_sec_ = 1.0;
  int num_candidates_ = 10;
  double sc_dist_threshold_ = 0.25;

  double ndt_resolution_ = 1.0;
  double ndt_step_size_ = 0.1;
  double ndt_trans_eps_ = 0.01;
  int ndt_max_iter_ = 40;
  int ndt_num_threads_ = 4;
  int ndt_neighborhood_ = 2;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "sc_ndt_initializer_node");
  SCNDTInitializer node;
  ros::spin();
  return 0;
}
