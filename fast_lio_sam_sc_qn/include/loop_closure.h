#ifndef FAST_LIO_SAM_SC_QN_LOOP_CLOSURE_H
#define FAST_LIO_SAM_SC_QN_LOOP_CLOSURE_H

///// C++ common headers
#include <tuple>
#include <vector>
#include <memory>
#include <limits>
#include <iostream>
#include <algorithm>
#include <utility> // pair
///// PCL
#include <pcl/point_types.h> //pt
#include <pcl/point_cloud.h> //cloud
///// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
///// Eigen
#include <Eigen/Eigen>
///// Nano-GICP
#include <nano_gicp/point_type_nano_gicp.hpp>
#include <nano_gicp/nano_gicp.hpp>
///// Quatro
#include <quatro/quatro_module.h>
///// ScanContext
#include <Scancontext.h>
///// coded headers
#include "pose_pcd.hpp"
#include "utilities.hpp"
using PcdPair = std::tuple<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>;

struct NanoGICPConfig
{
    int nano_thread_number_ = 0;
    int nano_correspondences_number_ = 15;
    int nano_max_iter_ = 32;
    int nano_ransac_max_iter_ = 5;
    double max_corr_dist_ = 2.0;
    double icp_score_thr_ = 10.0;
    double transformation_epsilon_ = 0.01;
    double euclidean_fitness_epsilon_ = 0.01;
    double ransac_outlier_rejection_threshold_ = 1.0;
};

struct QuatroConfig
{
    bool use_optimized_matching_ = true;
    bool estimat_scale_ = false;
    int quatro_max_num_corres_ = 500;
    int quatro_max_iter_ = 50;
    double quatro_distance_threshold_ = 30.0;
    double fpfh_normal_radius_ = 0.30; // It should be 2.5 - 3.0 * `voxel_res`
    double fpfh_radius_ = 0.50;        // It should be 5.0 * `voxel_res`
    double noise_bound_ = 0.30;
    double rot_gnc_factor_ = 1.40;
    double rot_cost_diff_thr_ = 0.0001;
};

struct LoopClosureConfig
{
    std::string mode_ = "scancontext";
    bool enable_quatro_ = true;
    bool enable_submap_matching_ = true;
    int num_submap_keyframes_ = 10;
    double voxel_res_ = 0.1;
    double scancontext_max_correspondence_distance_;
    NanoGICPConfig gicp_config_;
    QuatroConfig quatro_config_;
    int max_features_ = 1000;
    double ratio_test_thr_ = 0.75;
    int min_good_matches_ = 30;
    int min_inliers_ = 20;
    double min_inlier_ratio_ = 0.30;
    int temporal_exclusion_frames_ = 30;
};

struct RegistrationOutput
{
    bool is_valid_ = false;
    bool is_converged_ = false;
    double score_ = std::numeric_limits<double>::max();
    Eigen::Matrix4d pose_between_eig_ = Eigen::Matrix4d::Identity();
};

class LoopClosure
{
private:
    SCManager sc_manager_;
    nano_gicp::NanoGICP<PointType, PointType> nano_gicp_;
    std::shared_ptr<quatro<PointType>> quatro_handler_ = nullptr;
    int closest_keyframe_idx_ = -1;
    pcl::PointCloud<PointType>::Ptr src_cloud_;
    pcl::PointCloud<PointType>::Ptr dst_cloud_;
    pcl::PointCloud<PointType> coarse_aligned_;
    pcl::PointCloud<PointType> aligned_;
    cv::Ptr<cv::ORB> orb_;
    cv::Mat loop_match_image_;
    double last_loop_match_score_ = 0.0;
    LoopClosureConfig config_;

public:
    explicit LoopClosure(const LoopClosureConfig &config);
    ~LoopClosure();
    void updateScancontext(pcl::PointCloud<PointType> cloud);
    void computeVisualFeatures(PosePcd &keyframe);
    int fetchCandidateKeyframeIdx(const PosePcd &query_keyframe,
                                  const std::vector<PosePcd> &keyframes);
    int fetchCandidateKeyframeIdxByImage(const PosePcd &query_keyframe,
                                         const std::vector<PosePcd> &keyframes);
    PcdPair setSrcAndDstCloud(const std::vector<PosePcd> &keyframes,
                              const int src_idx,
                              const int dst_idx,
                              const int submap_range,
                              const double voxel_res,
                              const bool enable_quatro,
                              const bool enable_submap_matching);
    RegistrationOutput icpAlignment(const pcl::PointCloud<PointType> &src,
                                    const pcl::PointCloud<PointType> &dst);
    RegistrationOutput coarseToFineAlignment(const pcl::PointCloud<PointType> &src,
                                             const pcl::PointCloud<PointType> &dst);
    RegistrationOutput performLoopClosure(const PosePcd &query_keyframe,
                                          const std::vector<PosePcd> &keyframes,
                                          const int closest_keyframe_idx);
    pcl::PointCloud<PointType> getSourceCloud();
    pcl::PointCloud<PointType> getTargetCloud();
    pcl::PointCloud<PointType> getCoarseAlignedCloud();
    pcl::PointCloud<PointType> getFinalAlignedCloud();
    cv::Mat getLoopMatchImage();
    double getLastLoopMatchScore() const;
    int getClosestKeyframeidx();
};

#endif
