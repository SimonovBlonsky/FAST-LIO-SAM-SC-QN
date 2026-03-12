#include "loop_closure.h"

LoopClosure::LoopClosure(const LoopClosureConfig &config)
{
    config_ = config;
    const auto &gc = config_.gicp_config_;
    const auto &qc = config_.quatro_config_;
    ////// nano_gicp init
    nano_gicp_.setNumThreads(gc.nano_thread_number_);
    nano_gicp_.setCorrespondenceRandomness(gc.nano_correspondences_number_);
    nano_gicp_.setMaximumIterations(gc.nano_max_iter_);
    nano_gicp_.setRANSACIterations(gc.nano_ransac_max_iter_);
    nano_gicp_.setMaxCorrespondenceDistance(gc.max_corr_dist_);
    nano_gicp_.setTransformationEpsilon(gc.transformation_epsilon_);
    nano_gicp_.setEuclideanFitnessEpsilon(gc.euclidean_fitness_epsilon_);
    nano_gicp_.setRANSACOutlierRejectionThreshold(gc.ransac_outlier_rejection_threshold_);
    ////// quatro init
    quatro_handler_ = std::make_shared<quatro<PointType>>(qc.fpfh_normal_radius_,
                                                          qc.fpfh_radius_,
                                                          qc.noise_bound_,
                                                          qc.rot_gnc_factor_,
                                                          qc.rot_cost_diff_thr_,
                                                          qc.quatro_max_iter_,
                                                          qc.estimat_scale_,
                                                          qc.use_optimized_matching_,
                                                          qc.quatro_distance_threshold_,
                                                          qc.quatro_max_num_corres_);
    orb_ = cv::ORB::create(config_.max_features_);
    src_cloud_.reset(new pcl::PointCloud<PointType>);
    dst_cloud_.reset(new pcl::PointCloud<PointType>);
}

LoopClosure::~LoopClosure() {}

void LoopClosure::updateScancontext(pcl::PointCloud<PointType> cloud)
{
    sc_manager_.makeAndSaveScancontextAndKeys(cloud);
}

void LoopClosure::computeVisualFeatures(PosePcd &keyframe)
{
    keyframe.keypoints_.clear();
    keyframe.descriptors_.release();
    if (!keyframe.has_image_ || keyframe.img_.empty())
    {
        return;
    }

    cv::Mat gray;
    if (keyframe.img_.channels() == 3)
    {
        cv::cvtColor(keyframe.img_, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = keyframe.img_;
    }
    orb_->detectAndCompute(gray, cv::noArray(), keyframe.keypoints_, keyframe.descriptors_);
}

int LoopClosure::fetchCandidateKeyframeIdx(const PosePcd &query_keyframe,
                                           const std::vector<PosePcd> &keyframes)
{
    if (config_.mode_ == "visual")
    {
        return fetchCandidateKeyframeIdxByImage(query_keyframe, keyframes);
    }
    // from ScanContext, get the loop candidate
    std::pair<int, float> sc_detected_ = sc_manager_.detectLoopClosureIDGivenScan(query_keyframe.pcd_); // int: nearest node index,
                                                                                                        // float: relative yaw
    int candidate_keyframe_idx = sc_detected_.first;
    if (candidate_keyframe_idx >= 0) // if exists
    {
        // if close enough
        if ((keyframes[candidate_keyframe_idx].pose_corrected_eig_.block<3, 1>(0, 3) - query_keyframe.pose_corrected_eig_.block<3, 1>(0, 3))
                .norm() < config_.scancontext_max_correspondence_distance_)
        {
            return candidate_keyframe_idx;
        }
    }
    return -1;
}

int LoopClosure::fetchCandidateKeyframeIdxByImage(const PosePcd &query_keyframe,
                                                  const std::vector<PosePcd> &keyframes)
{
    loop_match_image_.release();
    last_loop_match_score_ = 0.0;
    if (!query_keyframe.has_image_ || query_keyframe.descriptors_.empty())
    {
        ROS_WARN("[Visual Loop] Query keyframe has no valid image descriptors");
        return -1;
    }

    int best_idx = -1;
    int best_inliers = 0;
    double best_ratio = 0.0;
    cv::Mat best_vis;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    const int max_candidate_idx = static_cast<int>(keyframes.size()) - config_.temporal_exclusion_frames_;
    for (int i = 0; i < max_candidate_idx; ++i)
    {
        const auto &candidate = keyframes[i];
        if (!candidate.has_image_ || candidate.descriptors_.empty())
        {
            continue;
        }

        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(query_keyframe.descriptors_, candidate.descriptors_, knn_matches, 2);

        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(knn_matches.size());
        for (const auto &match_pair : knn_matches)
        {
            if (match_pair.size() < 2) continue;
            if (match_pair[0].distance < config_.ratio_test_thr_ * match_pair[1].distance)
            {
                good_matches.push_back(match_pair[0]);
            }
        }

        int inliers = 0;
        double inlier_ratio = 0.0;
        std::vector<char> inlier_mask;
        if (static_cast<int>(good_matches.size()) >= config_.min_good_matches_)
        {
            std::vector<cv::Point2f> query_pts;
            std::vector<cv::Point2f> candidate_pts;
            query_pts.reserve(good_matches.size());
            candidate_pts.reserve(good_matches.size());
            for (const auto &match : good_matches)
            {
                query_pts.push_back(query_keyframe.keypoints_[match.queryIdx].pt);
                candidate_pts.push_back(candidate.keypoints_[match.trainIdx].pt);
            }

            cv::Mat inlier_mask_mat;
            cv::findHomography(query_pts, candidate_pts, cv::RANSAC, 3.0, inlier_mask_mat);
            if (!inlier_mask_mat.empty())
            {
                inlier_mask.assign(inlier_mask_mat.begin<uchar>(), inlier_mask_mat.end<uchar>());
                inliers = static_cast<int>(std::count(inlier_mask.begin(), inlier_mask.end(), static_cast<char>(1)));
                inlier_ratio = good_matches.empty() ? 0.0 : static_cast<double>(inliers) / static_cast<double>(good_matches.size());
            }
        }

        ROS_INFO("[Visual Loop] query %d vs %d: good_matches=%zu, inliers=%d, inlier_ratio=%.3f",
                 query_keyframe.idx_, candidate.idx_, good_matches.size(), inliers, inlier_ratio);

        if (inliers >= config_.min_inliers_ && inlier_ratio >= config_.min_inlier_ratio_)
        {
            if (inliers > best_inliers || (inliers == best_inliers && inlier_ratio > best_ratio))
            {
                best_idx = i;
                best_inliers = inliers;
                best_ratio = inlier_ratio;
                cv::drawMatches(query_keyframe.img_, query_keyframe.keypoints_,
                                candidate.img_, candidate.keypoints_,
                                good_matches, best_vis,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                inlier_mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            }
        }
    }

    if (best_idx >= 0)
    {
        loop_match_image_ = best_vis;
        last_loop_match_score_ = static_cast<double>(best_inliers);
    }
    return best_idx;
}

PcdPair LoopClosure::setSrcAndDstCloud(const std::vector<PosePcd> &keyframes,
                                       const int src_idx,
                                       const int dst_idx,
                                       const int submap_range,
                                       const double voxel_res,
                                       const bool enable_quatro,
                                       const bool enable_submap_matching)
{
    pcl::PointCloud<PointType> dst_accum, src_accum;
    int num_approx = keyframes[src_idx].pcd_.size() * 2 * submap_range;
    src_accum.reserve(num_approx);
    dst_accum.reserve(num_approx);
    if (enable_submap_matching)
    {
        for (int i = src_idx - submap_range; i < src_idx + submap_range + 1; ++i)
        {
            if (i >= 0 && i < static_cast<int>(keyframes.size() - 1))
            {
                src_accum += transformPcd(keyframes[i].pcd_, keyframes[i].pose_corrected_eig_);
            }
        }
        for (int i = dst_idx - submap_range; i < dst_idx + submap_range + 1; ++i)
        {
            if (i >= 0 && i < static_cast<int>(keyframes.size() - 1))
            {
                dst_accum += transformPcd(keyframes[i].pcd_, keyframes[i].pose_corrected_eig_);
            }
        }
    }
    else
    {
        src_accum = transformPcd(keyframes[src_idx].pcd_, keyframes[src_idx].pose_corrected_eig_);
        if (enable_quatro)
        {
            dst_accum = transformPcd(keyframes[dst_idx].pcd_, keyframes[dst_idx].pose_corrected_eig_);
        }
        else
        {
            // For ICP matching,
            // empirically scan-to-submap matching works better
            for (int i = dst_idx - submap_range; i < dst_idx + submap_range + 1; ++i)
            {
                if (i >= 0 && i < static_cast<int>(keyframes.size() - 1))
                {
                    dst_accum += transformPcd(keyframes[i].pcd_, keyframes[i].pose_corrected_eig_);
                }
            }
        }
    }
    return {*voxelizePcd(src_accum, voxel_res), *voxelizePcd(dst_accum, voxel_res)};
}

RegistrationOutput LoopClosure::icpAlignment(const pcl::PointCloud<PointType> &src,
                                             const pcl::PointCloud<PointType> &dst)
{
    RegistrationOutput reg_output;
    aligned_.clear();
    // merge subkeyframes before ICP
    pcl::PointCloud<PointType>::Ptr src_cloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr dst_cloud(new pcl::PointCloud<PointType>());
    *src_cloud = src;
    *dst_cloud = dst;
    nano_gicp_.setInputSource(src_cloud);
    nano_gicp_.calculateSourceCovariances();
    nano_gicp_.setInputTarget(dst_cloud);
    nano_gicp_.calculateTargetCovariances();
    nano_gicp_.align(aligned_);

    // handle results
    reg_output.score_ = nano_gicp_.getFitnessScore();
    // if matchness score is lower than threshold, (lower is better)
    if (nano_gicp_.hasConverged() && reg_output.score_ < config_.gicp_config_.icp_score_thr_)
    {
        reg_output.is_valid_ = true;
        reg_output.is_converged_ = true;
        reg_output.pose_between_eig_ = nano_gicp_.getFinalTransformation().cast<double>();
    }
    return reg_output;
}

RegistrationOutput LoopClosure::coarseToFineAlignment(const pcl::PointCloud<PointType> &src,
                                                      const pcl::PointCloud<PointType> &dst)
{
    RegistrationOutput reg_output;
    coarse_aligned_.clear();

    reg_output.pose_between_eig_ = (quatro_handler_->align(src, dst, reg_output.is_converged_));
    if (!reg_output.is_converged_)
    {
        return reg_output;
    }
    else // if valid,
    {
        // coarse align with the result of Quatro
        coarse_aligned_ = transformPcd(src, reg_output.pose_between_eig_);
        const auto &fine_output = icpAlignment(coarse_aligned_, dst);
        const auto quatro_tf_ = reg_output.pose_between_eig_;
        reg_output = fine_output;
        reg_output.pose_between_eig_ = fine_output.pose_between_eig_ * quatro_tf_;
    }
    return reg_output;
}

RegistrationOutput LoopClosure::performLoopClosure(const PosePcd &query_keyframe,
                                                   const std::vector<PosePcd> &keyframes,
                                                   const int closest_keyframe_idx)
{
    RegistrationOutput reg_output;
    closest_keyframe_idx_ = closest_keyframe_idx;
    if (closest_keyframe_idx_ >= 0)
    {
        // Quatro + NANO-GICP to check loop (from front_keyframe to closest keyframe's neighbor)
        const auto &[src_cloud, dst_cloud] = setSrcAndDstCloud(keyframes,
                                                               query_keyframe.idx_,
                                                               closest_keyframe_idx_,
                                                               config_.num_submap_keyframes_,
                                                               config_.voxel_res_,
                                                               config_.enable_quatro_,
                                                               config_.enable_submap_matching_);
        // Only for visualization
        *src_cloud_ = src_cloud;
        *dst_cloud_ = dst_cloud;

        if (config_.enable_quatro_)
        {
            std::cout << "\033[1;35mExecute coarse-to-fine alignment: " << src_cloud.size()
                      << " vs " << dst_cloud.size() << "\033[0m\n";
            return coarseToFineAlignment(src_cloud, dst_cloud);
        }
        else
        {
            std::cout << "\033[1;35mExecute GICP: " << src_cloud.size() << " vs "
                      << dst_cloud.size() << "\033[0m\n";
            return icpAlignment(src_cloud, dst_cloud);
        }
    }
    else
    {
        return reg_output; // dummy output whose `is_valid` is false
    }
}

pcl::PointCloud<PointType> LoopClosure::getSourceCloud()
{
    return *src_cloud_;
}

pcl::PointCloud<PointType> LoopClosure::getTargetCloud()
{
    return *dst_cloud_;
}

pcl::PointCloud<PointType> LoopClosure::getCoarseAlignedCloud()
{
    return coarse_aligned_;
}

// NOTE(hlim): To cover ICP-only mode, I just set `Final`, not `Fine`
pcl::PointCloud<PointType> LoopClosure::getFinalAlignedCloud()
{
    return aligned_;
}

cv::Mat LoopClosure::getLoopMatchImage()
{
    return loop_match_image_;
}

double LoopClosure::getLastLoopMatchScore() const
{
    return last_loop_match_score_;
}

int LoopClosure::getClosestKeyframeidx()
{
    return closest_keyframe_idx_;
}
