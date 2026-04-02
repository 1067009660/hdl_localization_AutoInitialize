#include "scancontext_init_localizer/scan_context.hpp"

#include <Eigen/Geometry>
#include <boost/filesystem.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace scancontext_init_localizer {

namespace {
constexpr float kEmptyValue = -1000.0f;
}  // namespace

ScanContext::ScanContext(const Params& params) : params_(params) {}

int ScanContext::xy2ring(double x, double y) const {
  const double r = std::sqrt(x * x + y * y);
  if (r > params_.max_radius) {
    return -1;
  }
  const double ring_step = params_.max_radius / static_cast<double>(params_.num_rings);
  const int ring = std::min(params_.num_rings - 1, static_cast<int>(r / ring_step));
  return ring;
}

int ScanContext::xy2sector(double x, double y) const {
  double theta = std::atan2(y, x);
  if (theta < 0.0) {
    theta += 2.0 * M_PI;
  }
  const double sector_step = 2.0 * M_PI / static_cast<double>(params_.num_sectors);
  return std::min(params_.num_sectors - 1, static_cast<int>(theta / sector_step));
}

Eigen::MatrixXf ScanContext::makeDescriptor(const pcl::PointCloud<pcl::PointXYZI>& cloud) const {
  Eigen::MatrixXf desc = Eigen::MatrixXf::Constant(params_.num_rings, params_.num_sectors, kEmptyValue);
  for (const auto& pt : cloud.points) {
    const int ring = xy2ring(pt.x, pt.y);
    if (ring < 0) {
      continue;
    }
    const int sector = xy2sector(pt.x, pt.y);
    const float z = pt.z + static_cast<float>(params_.lidar_height);
    if (z > desc(ring, sector)) {
      desc(ring, sector) = z;
    }
  }

  for (int r = 0; r < desc.rows(); ++r) {
    for (int c = 0; c < desc.cols(); ++c) {
      if (desc(r, c) < -100.0f) {
        desc(r, c) = 0.0f;
      }
    }
  }
  return desc;
}

Eigen::VectorXf ScanContext::makeRingKey(const Eigen::MatrixXf& descriptor) const {
  Eigen::VectorXf key(descriptor.rows());
  for (int r = 0; r < descriptor.rows(); ++r) {
    key(r) = descriptor.row(r).mean();
  }
  return key;
}

Eigen::MatrixXf ScanContext::circShift(const Eigen::MatrixXf& mat, int shift) const {
  const int cols = mat.cols();
  const int normalized_shift = ((shift % cols) + cols) % cols;
  if (normalized_shift == 0) {
    return mat;
  }

  Eigen::MatrixXf out(mat.rows(), mat.cols());
  out.leftCols(normalized_shift) = mat.rightCols(normalized_shift);
  out.rightCols(cols - normalized_shift) = mat.leftCols(cols - normalized_shift);
  return out;
}

float ScanContext::cosineDistanceColumnwise(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) const {
  double sum_sim = 0.0;
  int valid_cols = 0;
  for (int c = 0; c < a.cols(); ++c) {
    const Eigen::VectorXf ca = a.col(c);
    const Eigen::VectorXf cb = b.col(c);
    const float na = ca.norm();
    const float nb = cb.norm();
    if (na < 1e-6f || nb < 1e-6f) {
      continue;
    }
    sum_sim += static_cast<double>(ca.dot(cb) / (na * nb));
    ++valid_cols;
  }

  if (valid_cols == 0) {
    return 1.0f;
  }
  const double mean_sim = sum_sim / static_cast<double>(valid_cols);
  return static_cast<float>(1.0 - mean_sim);
}

std::pair<float, int> ScanContext::distanceWithYaw(const Eigen::MatrixXf& query,
                                                   const Eigen::MatrixXf& target) const {
  float best_dist = std::numeric_limits<float>::max();
  int best_shift = 0;

  for (int shift = 0; shift < params_.num_sectors; ++shift) {
    const Eigen::MatrixXf shifted = circShift(query, shift);
    const float dist = cosineDistanceColumnwise(shifted, target);
    if (dist < best_dist) {
      best_dist = dist;
      best_shift = shift;
    }
  }

  return {best_dist, best_shift};
}

float ScanContext::ringKeyDistance(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::max();
  }
  return (a - b).norm();
}

double ScanContext::sectorAngleRad() const {
  return 2.0 * M_PI / static_cast<double>(params_.num_sectors);
}

bool saveDatabaseEntry(const std::string& root_dir, const DatabaseEntry& entry) {
  const boost::filesystem::path root(root_dir);
  const boost::filesystem::path entries_dir = root / "entries";
  boost::filesystem::create_directories(entries_dir);

  std::ostringstream fname;
  fname << std::setw(6) << std::setfill('0') << entry.index << ".sc";
  const boost::filesystem::path file_path = entries_dir / fname.str();

  std::ofstream ofs(file_path.string());
  if (!ofs.is_open()) {
    return false;
  }

  ofs << entry.index << " "
      << entry.position.x() << " " << entry.position.y() << " " << entry.position.z() << " "
      << entry.orientation.x() << " " << entry.orientation.y() << " "
      << entry.orientation.z() << " " << entry.orientation.w() << " "
      << entry.pcd_path << "\n";

  for (int i = 0; i < entry.ring_key.size(); ++i) {
    ofs << entry.ring_key(i);
    if (i + 1 < entry.ring_key.size()) {
      ofs << " ";
    }
  }
  ofs << "\n";

  ofs << entry.descriptor.rows() << " " << entry.descriptor.cols() << "\n";
  for (int r = 0; r < entry.descriptor.rows(); ++r) {
    for (int c = 0; c < entry.descriptor.cols(); ++c) {
      ofs << entry.descriptor(r, c);
      if (c + 1 < entry.descriptor.cols()) {
        ofs << " ";
      }
    }
    ofs << "\n";
  }

  std::ofstream index_ofs((root / "index.txt").string(), std::ios::app);
  if (!index_ofs.is_open()) {
    return false;
  }
  index_ofs << file_path.string() << "\n";
  return true;
}

bool loadDatabase(const std::string& root_dir,
                  int num_rings,
                  int num_sectors,
                  std::vector<DatabaseEntry>* entries,
                  std::string* error_msg) {
  entries->clear();

  std::ifstream ifs((boost::filesystem::path(root_dir) / "index.txt").string());
  if (!ifs.is_open()) {
    if (error_msg) {
      *error_msg = "failed to open index.txt";
    }
    return false;
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }

    std::ifstream efs(line);
    if (!efs.is_open()) {
      continue;
    }

    DatabaseEntry entry;
    efs >> entry.index
        >> entry.position.x() >> entry.position.y() >> entry.position.z()
        >> entry.orientation.x() >> entry.orientation.y()
        >> entry.orientation.z() >> entry.orientation.w()
        >> entry.pcd_path;

    entry.ring_key = Eigen::VectorXf(num_rings);
    for (int i = 0; i < num_rings; ++i) {
      efs >> entry.ring_key(i);
    }

    int rows = 0;
    int cols = 0;
    efs >> rows >> cols;
    if (rows != num_rings || cols != num_sectors) {
      continue;
    }

    entry.descriptor = Eigen::MatrixXf(rows, cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        efs >> entry.descriptor(r, c);
      }
    }

    entries->push_back(entry);
  }

  if (entries->empty()) {
    if (error_msg) {
      *error_msg = "database loaded but no valid entries";
    }
    return false;
  }

  return true;
}

}  // namespace scancontext_init_localizer
