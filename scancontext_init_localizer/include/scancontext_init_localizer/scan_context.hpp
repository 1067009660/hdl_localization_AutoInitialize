#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <string>
#include <vector>

namespace scancontext_init_localizer {

class ScanContext {
public:
  struct Params {
    int num_rings = 20;
    int num_sectors = 60;
    double max_radius = 80.0;
    double lidar_height = 2.0;
  };

  explicit ScanContext(const Params& params);

  Eigen::MatrixXf makeDescriptor(const pcl::PointCloud<pcl::PointXYZI>& cloud) const;
  Eigen::VectorXf makeRingKey(const Eigen::MatrixXf& descriptor) const;

  std::pair<float, int> distanceWithYaw(const Eigen::MatrixXf& query,
                                        const Eigen::MatrixXf& target) const;

  float ringKeyDistance(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;
  double sectorAngleRad() const;

private:
  Params params_;
  int xy2ring(double x, double y) const;
  int xy2sector(double x, double y) const;
  Eigen::MatrixXf circShift(const Eigen::MatrixXf& mat, int shift) const;
  float cosineDistanceColumnwise(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) const;
};

struct DatabaseEntry {
  int index = -1;
  std::string pcd_path;
  Eigen::VectorXf ring_key;
  Eigen::MatrixXf descriptor;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

bool saveDatabaseEntry(const std::string& root_dir, const DatabaseEntry& entry);
bool loadDatabase(const std::string& root_dir,
                  int num_rings,
                  int num_sectors,
                  std::vector<DatabaseEntry>* entries,
                  std::string* error_msg);

}  // namespace scancontext_init_localizer
