# hdl_localization_AutoInitialize
本工作空间提供两种基于激光雷达点云的初始位姿估计方法，用于为 `hdl_localization` 等定位系统提供自动重定位初值。两种方法均采用 **离线建库 + 在线检索** 的架构，与 Faster-LIO 建图流程无缝集成。基于ScanContext和BTC描述子自动初始化Hdl重定位算法的初始位姿，不用在Rviz中通过2D Pose Estimate手动初始化

## 🎬 演示效果

![自动初始化演示](./auto_initialize.gif)

*演示内容：通过服务调用自动计算初始位姿并发布到 `/initialpose` 话题*

## 📊 方法特性对比

| 特性 | Scan Context (SC) 方法 | BTC 描述子方法 |
|------|------------------------|----------------|
| **描述子类型** | 环-扇区距离矩阵 (2D) | 二进制三角形几何描述子 (3D) |
| **旋转不变性** | ✅ 通过循环移位实现 | ⚠️ 依赖三角形几何关系 |
| **匹配速度** | 快速（环键粗筛 + 矩阵匹配） | 中等（空间哈希 + 几何验证） |
| **内存占用** | 较小（仅保存描述子） | 较大（保存描述子+平面点云） |
| **适用场景** | 室外大场景、旋转变化大 | 结构化环境、平面特征丰富 |
| **视角依赖性** | 需要 360° 全向点云 | 对视角变化敏感 |
| **并行加速** | 未实现 | ✅ 支持 C++17 并行 STL |

## 🛠 安装与编译

### 环境要求
- ROS Noetic
- CMake ≥ 3.0.2
- C++17 编译器
- TBB (Intel Threading Building Blocks)

### 编译步骤
```bash
cd /your_workspace/src/hdl_localization_AutoInitialize
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## 🗺️ 与 Faster-LIO 建图集成

两种方法均设计为与 **Faster-LIO** 建图过程同步运行，实时构建描述子数据库。

### 同时运行建图与数据库构建
```bash
# 终端1: 启动 Faster-LIO 建图
# （根据您的 Faster-LIO 启动命令）

# 终端2: 启动 Scan Context 数据库构建
roslaunch scancontext_init_localizer sc_map_builder.launch

# 或终端2: 启动 BTC 数据库构建  
roslaunch btc_init_localizer btc_map_builder.launch
```

### 输入话题要求
两种方法均需要订阅以下话题：
- `/velodyne_points` (`sensor_msgs/PointCloud2`) - 原始激光雷达点云
- `/Odometry` (`nav_msgs/Odometry`) - 来自 Faster-LIO 的 map 坐标系下位姿

## 📁 离线建库

### Scan Context 建库
```bash
roslaunch scancontext_init_localizer sc_map_builder.launch
```

**输出目录结构** (`output_dir` 默认为 `scancontext_db`)：
```
index.txt
entries/*.sc     # SC 描述子 + 位姿元数据
clouds/*.pcd     # 历史关键帧点云
```

### BTC 描述子建库
```bash
roslaunch btc_init_localizer btc_map_builder.launch
```

**输出目录结构** (`output_dir` 默认为 `btc_db`)：
```
index.txt
entries/*.btc    # BTC 描述子 + 位姿元数据
clouds/*.pcd     # 历史关键帧点云
planes/*.pcd     # 历史平面点云（用于几何验证）
```

## 🔍 在线初始化服务

### Scan Context 初始化
```bash
# 启动服务节点
roslaunch scancontext_init_localizer sc_ndt_initializer.launch

# 调用初始化服务
rosservice call /scancontext_initialize
```

**处理流程**：
1. 从 `/velodyne_points` 获取当前帧点云
2. 在离线 SC 数据库中检索匹配历史帧
3. 加载匹配的历史点云及其位姿 `T_map_historical`
4. 执行 NDT_OMP 精配准，得到 `T_historical_current`
5. 计算当前位姿：`T_map_current = T_map_historical × T_historical_current`
6. 发布到 `/initialpose` 话题

### BTC 描述子初始化
```bash
# 启动服务节点
roslaunch btc_init_localizer btc_ndt_initializer.launch

# 调用初始化服务
rosservice call /btc_initialize
```

**处理流程**：
1. 从 `/velodyne_points` 获取当前帧点云
2. 提取 BTC 描述子并在数据库中检索匹配
3. 通过几何验证得到粗位姿 `T_historical_current`
4. 执行 NDT_OMP 精配准进行精细优化
5. 计算当前位姿：`T_map_current = T_map_historical × T_historical_current`
6. 发布到 `/initialpose` 话题

## ⚙️ 关键参数配置

### Scan Context 参数
```yaml
# sc_initializer.yaml
sc_distance_threshold: 0.25    # SC 匹配距离阈值，大于此值认为失败
num_candidates: 10             # 环键粗筛后保留的候选帧数
num_rings: 20                  # 径向环数（建议 20）
num_sectors: 60                # 角度扇区数（建议 60）
max_radius: 80.0               # 最大考虑半径（米）
```

### BTC 描述子参数
```yaml
# btc_initializer.yaml
btc_score_threshold: 0.15      # BTC 几何验证得分阈值
skip_near_num: -1              # 跳过时间邻近的帧（-1 表示不跳过）
similarity_threshold_: 0.7     # 二进制描述子相似度阈值
parallel_stl_enable: 1         # 启用并行加速
parallel_stl_min_size: 256     # 并行触发的最小数据量
```

### 共用 NDT 参数
```yaml
ndt_resolution: 1.0            # NDT 体素分辨率
ndt_step_size: 0.1             # 优化步长
ndt_trans_eps: 0.01            # 变换收敛阈值
ndt_max_iter: 40               # 最大迭代次数
ndt_num_threads: 4             # NDT 并行线程数
```

## ⚠️ 重要注意事项

### 1. 雷达视野要求
- **Scan Context**：需要 **360° 全向点云** 才能发挥旋转不变性优势
- **BTC 描述子**：对视角变化敏感，建议使用全向雷达或往返建图

### 2. 方向依赖性问题
如果建图时只采集了**单向**路径（如 A→B），则反向行走（B→A）时可能匹配失败：
- **原因**：180° 前向雷达在反向时看到完全不同的场景
- **解决方案**：
  - 使用 360° 旋转雷达
  - 进行往返建图（A→B 和 B→A 各建一次）
  - 调整匹配阈值（临时缓解）

### 3. 数据库规模与性能
- 数据库规模增大会增加检索时间
- BTC 方法已实现并行加速，可通过 `parallel_stl_enable` 控制
- 建议定期清理冗余或相似度过高的描述子

## 🔧 故障排除

### Q1: 服务调用返回 "BTC matching failed" 或 "SC match failed"
- 检查数据库路径配置是否正确
- 确认当前点云与建图时雷达配置一致
- 尝试降低匹配阈值（`btc_score_threshold` / `sc_distance_threshold`）

### Q2: 初始化位姿不准确
- 检查 NDT 参数是否适合当前场景
- 确认 Faster-LIO 建图时位姿是否准确
- 尝试调整 `ndt_resolution` 和 `ndt_step_size`

### Q3: 运行速度慢
- 启用并行加速（BTC 方法）
- 减少 `num_candidates`（SC 方法）
- 检查 CPU 占用，确认 TBB 库已正确安装

### Q4: 反向行走匹配失败
- 这是**预期行为**，因为两种方法都对视角敏感
- 解决方案见上文"方向依赖性问题"部分

## 📚 算法原理简介

### Scan Context (SC)
将点云投影到极坐标网格，计算每个环-扇区单元的最大高度值，形成 2D 矩阵描述子。通过循环移位比较处理不同朝向。

### BTC 描述子
从点云提取平面特征，构建空间三角形，计算二进制投影模式。通过空间哈希加速检索，结合几何验证提高匹配可靠性。

## 📚 参考与致谢

本工作空间的实现参考了以下开源项目：

1. **[hdl_localization](https://github.com/koide3/hdl_localization)** - 提供了基于 NDT 的激光雷达定位框架，本项目的初始位姿估计结果可直接用于 `hdl_localization` 的初始化输入。

2. **[Voxel-SLAM](https://github.com/hku-mars/Voxel-SLAM)** - 提供了 BTC (Binary Triangle Context) 描述子的实现思路，本项目的 BTC 描述子构建与匹配算法参考了该仓库的算法设计。

3. **[faster-lio](https://github.com/gaoxiang12/faster-lio)** - 提供了高效激光雷达惯性里程计建图框架，本项目与 Faster-LIO 建图流程无缝集成，实时构建描述子数据库。

感谢以上开源项目作者对社区作出的贡献。

---

**最后更新**: 2025年4月2日  
**兼容系统**: ROS Noetic, Ubuntu 20.04  
**测试平台**: Faster-LIO + Velodyne 雷达
