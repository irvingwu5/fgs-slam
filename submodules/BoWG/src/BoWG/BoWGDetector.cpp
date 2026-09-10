// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: BoWGDetector.cpp
 * Date: August 2024
 * Author: Xiang Fei
 * Description: A loop closure detector using BoWG, an example usage is shown in demo.cpp 
 * When you want to integrate BoWG into other SLAM systems
 * I strongly recommend you to read this file
 *
 */


#include "BoWGDetector.h"

#include <algorithm>
#include <stdexcept>

namespace BoWG{

// 函数作用：执行 BoWGDetector 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BoWGDetector::BoWGDetector(void) : BoWGDetector(false, 0) {}

// 函数作用：执行 BoWGDetector 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - use_gui：use_gui 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - delay_ms：delay_ms 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BoWGDetector::BoWGDetector(bool use_gui, int delay_ms)
    : db_wg(WG_WEIGHT_TYPE, WG_SCORING_TYPE, W_SCORING_TYPE),
      enable_gui(use_gui),
      frame_delay_ms(delay_ms),
      is_paused(false)
{
    voc.reset(new BriefVocabulary(VOCABULARY_PATH));
    switch(W_WEIGHT_TYPE)
    {
        case DBoW2::TF_IDF:
            voc->setWeightingType(DBoW2::TF_IDF);
            break;
        case DBoW2::TF:
            voc->setWeightingType(DBoW2::TF);
            break;
        case DBoW2::IDF:
            voc->setWeightingType(DBoW2::IDF);
            break;
        case DBoW2::BINARY:
            voc->setWeightingType(DBoW2::BINARY);
            break;
    }
    
    switch(W_SCORING_TYPE)
    {
        case DBoW2::L1_NORM:
            voc->setScoringType(DBoW2::L1_NORM);
            break;
        case DBoW2::L2_NORM:
            voc->setScoringType(DBoW2::L2_NORM);
            break;
        case DBoW2::CHI_SQUARE:
            voc->setScoringType(DBoW2::CHI_SQUARE);
            break;
        case DBoW2::KL:
            voc->setScoringType(DBoW2::KL);
            break;
        case DBoW2::BHATTACHARYYA:
            voc->setScoringType(DBoW2::BHATTACHARYYA);
            break;
        case DBoW2::DOT_PRODUCT:
            voc->setScoringType(DBoW2::DOT_PRODUCT);
            break;
    }
    
    if (GEOM_DI)
        db.setVocabulary(*voc, true, DI_LEVEL);
    else
        db.setVocabulary(*voc, false, 0);

    db_wg.min_prev_score = MIN_PREV_W_SCORE;
    db_wg.min_prev_wg_score = MIN_PREV_WG_SCORE;
    db_wg.min_prev_dist_score = MIN_PREV_DIST_SCORE;

    if (enable_gui) {
        initializeGUI();
    }
}

// 函数作用：释放对象持有的资源。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BoWGDetector::~BoWGDetector(void)
{
}

size_t BoWGDetector::size() const
{
    return db.size();
}

void BoWGDetector::reset()
{
    db.clear();
    db_wg = BoWGDatabase(WG_WEIGHT_TYPE, WG_SCORING_TYPE, W_SCORING_TYPE);
    db_wg.min_prev_score = MIN_PREV_W_SCORE;
    db_wg.min_prev_wg_score = MIN_PREV_WG_SCORE;
    db_wg.min_prev_dist_score = MIN_PREV_DIST_SCORE;
    map_des.clear();
    map_kpts.clear();
    m_image_descriptors.clear();
    m_image_keys.clear();
    m_image_wg_keys.clear();
    display_image.release();
    is_paused = false;
}

DetectionResult BoWGDetector::queryThenAdd(
    const cv::Mat& image_bgr, int top_k, int temporal_exclusion)
{
    return processImage(
        image_bgr, true, top_k, temporal_exclusion, false, nullptr);
}

void BoWGDetector::image_detect(
    cv::Mat image, bool is_query, int size, std::vector<int>& cor_row)
{
    const int top_k = std::max(1, size - DISLOCAL);
    processImage(image, is_query, top_k, DISLOCAL, true, &cor_row);
}

// 函数作用：执行 updateRowVector 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - row_vec：row_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - indices：indices 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGDetector::updateRowVector(std::vector<int>& row_vec, const std::vector<int>& indices)
{
    for (int idx : indices) {
        if (idx >= 0 && idx < static_cast<int>(row_vec.size())) {
            row_vec[idx] = 1;
        } else {
            std::cerr << "Index " << idx << " is out of bounds for row_vec of size " 
                      << row_vec.size() << std::endl;
        }
    }
}


// 函数作用：提取图像特征、查询历史数据库并输出回环候选。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - image：输入图像。
//   - is_query：is_query 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - size：size 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - cor_row：cor_row 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
DetectionResult BoWGDetector::processImage(
    const cv::Mat& image,
    bool should_query,
    int top_k,
    int temporal_exclusion,
    bool run_legacy_acceptance,
    std::vector<int>* cor_row)
{
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("image_bgr must be a non-empty CV_8UC3 image");
    }
    if (top_k < 1) {
        throw std::invalid_argument("top_k must be at least 1");
    }
    if (temporal_exclusion < 1) {
        throw std::invalid_argument("temporal_exclusion must be at least 1");
    }

    const std::chrono::steady_clock::time_point extract_start =
        std::chrono::steady_clock::now();
    if (enable_gui) {
        char key = cv::waitKey(1);
        if (key == 'p' || key == 'P') {
            togglePause();
        }
        waitIfPaused();
    }

    std::vector<cv::KeyPoint> keypoints;

    // center point
    cv::Point2f center_pt=cv::Point2f(image.cols/2,image.rows/2);

    // ItemID of the keyframe
    BoWG::ItemID kf_iid = db_wg.get_itemId();
    db_wg.cur_image_id = static_cast<int>(kf_iid);

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    if (enable_gui) {
        char key = cv::waitKey(1);
        if (key == 'p' || key == 'P') {
            togglePause();
        }
        waitIfPaused();
    }

    std::vector<BRIEF::bitset> brief_descriptors;
    if (!MS_FEATURE) {
        // use fast
        cv::FAST(grayImage, keypoints, 10, true);
        cv::KeyPointsFilter::retainBest(keypoints, FEATURE_CNT);
        DVision::BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
        extractor(grayImage, keypoints, brief_descriptors);
    }
    else {
        // use multi-scale fast
        std::vector<cv::Mat> gray_pyramid; 
        gray_pyramid.push_back(grayImage);
        for (int i = 0; i < MS_LEVELS - 1; i++) // use Gaussian Pyramid to detect multi-scale features
        {
            cv::Mat down_gray_image; // downsampled image
            cv::pyrDown(gray_pyramid.back(),down_gray_image);
            gray_pyramid.push_back(down_gray_image);
        }

        DVision::BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
        for (int i = 0; i < MS_LEVELS; i++)
        {
            std::vector<cv::KeyPoint> tmp_keypts;
            cv::FAST(gray_pyramid[i], tmp_keypts, 10, true);
            cv::KeyPointsFilter::retainBest(tmp_keypts, FEATURE_CNT / std::pow(2, i));
            for (int j = 0; j < (int)tmp_keypts.size(); j++)
            {
                tmp_keypts[j].size = FEATURE_SIZE * pow(2,i);
                cv::KeyPoint key = tmp_keypts[j];
                key.pt.x = key.pt.x * pow(2,i); // scale the coordinates of points to the original image size
                key.pt.y = key.pt.y * pow(2,i);
                keypoints.push_back(key);
            }
            std::vector<BRIEF::bitset> tmp_descriptors;
            extractor(gray_pyramid[i], tmp_keypts, tmp_descriptors);
            brief_descriptors.insert(brief_descriptors.end(),tmp_descriptors.begin(),tmp_descriptors.end());
        }
    }

    if (enable_gui) {
        char key = cv::waitKey(1);
        if (key == 'p' || key == 'P') {
            togglePause();
        }
        waitIfPaused();
    }

    m_image_descriptors.push_back(brief_descriptors);
    m_image_keys.push_back(keypoints);
    map_kpts[kf_iid] = keypoints;
    map_des[kf_iid] = brief_descriptors;

    // obtain the bow vector of the current image
    DBoW2::BowVector cur_bow_vec;
    voc->transform(brief_descriptors, cur_bow_vec);

    std::cout << "word num: " << keypoints.size() << std::endl;

    std::vector<int> in_kernel_vec;
    if (USE_DISTRIBUTION) {
        // obtain the input of kernel density estimation
        in_kernel_vec = db_wg.kernel_input(keypoints,center_pt,DISTRIBUTION_BATCH);
    }

    // keyframe word groups
    std::vector<WordGroupID> kf_wgs;
    std::vector<cv::KeyPoint> wg_keypoints;
    if (USE_WG) {
        for (int i = 0; i < (int)keypoints.size(); i++)
        {
            cv::KeyPoint tmp_kp1 = keypoints[i];
            for (int j = i + 1; j < (int)keypoints.size(); j++)
            {
                cv::KeyPoint tmp_kp2 = keypoints[j];
                float distance = cv::norm(tmp_kp1.pt - tmp_kp2.pt);
                if (distance < tmp_kp1.size)
                {
                    //detect multi-scale feature and word pair
                    DBoW2::WordId wid1 = voc->transform(brief_descriptors[i]);
                    BoWG::WordGroupID wg_id = db_wg.queryWordGroup(wid1);
                    kf_wgs.push_back(wg_id);
                    wg_keypoints.push_back(tmp_kp1);
                }
                if (distance < tmp_kp2.size)
                {
                    DBoW2::WordId wid2 = voc->transform(brief_descriptors[j]);
                    BoWG::WordGroupID wg_id = db_wg.queryWordGroup(wid2);
                    kf_wgs.push_back(wg_id);
                    wg_keypoints.push_back(tmp_kp2);
                }
            }
        }

        std::cout << "word group num: " << kf_wgs.size() << std::endl; //total number of word groups in this image
        
        // sort kf_wgs for computing the BoWGVector of the keyframe
        std::sort(kf_wgs.begin(), kf_wgs.end());
    }

    m_image_wg_keys.push_back(wg_keypoints);
    BoWG::BoWGVector kf_bowgVector = db_wg.computeBoWGVector(kf_wgs);

    DBoW2::FeatureVector featvec;
    if (GEOM_DI) {
        voc->transform(brief_descriptors, cur_bow_vec, featvec, DI_LEVEL);
    }

    const double extract_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - extract_start).count();
    DBoW2::QueryResults ret;
    double query_ms = 0.0;
    const int max_id = static_cast<int>(kf_iid) - temporal_exclusion;
    if (should_query && max_id > 0)
    {
        if (enable_gui) {
            char key = cv::waitKey(1);
            if (key == 'p' || key == 'P') {
                togglePause();
            }
            waitIfPaused();
        }

        DBoW2::QueryResults in_ret;
        const std::chrono::steady_clock::time_point query_start =
            std::chrono::steady_clock::now();
        db.query(brief_descriptors, in_ret, top_k, max_id);

        if (USE_WG && USE_DISTRIBUTION) {
            db_wg.query_bowg(in_ret, ret, cur_bow_vec, kf_bowgVector, in_kernel_vec, top_k, max_id,
                    W_WEIGHT, WG_WEIGHT, USE_TEMPORAL_SCORE, PREV_WEIGHT_TH, TEMPORAL_PARAM);
        }
        else if (USE_WG) {
            db_wg.query_bowg(in_ret, ret, cur_bow_vec, kf_bowgVector, top_k, max_id,
                    W_WEIGHT, USE_TEMPORAL_SCORE, PREV_WEIGHT_TH, TEMPORAL_PARAM);
        }
        else {
            db_wg.query_words(in_ret, ret, cur_bow_vec, top_k, USE_TEMPORAL_SCORE, PREV_WEIGHT_TH, TEMPORAL_PARAM);
        }

        query_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - query_start).count();

        db_wg.res_table[kf_iid] = ret;

        if (GEOM_DI)
            db.add(cur_bow_vec, featvec);
        else
            db.add(brief_descriptors);
            

        db_wg.add(kf_bowgVector, kf_iid);

        db_wg.prev_bow_vec = cur_bow_vec;
        db_wg.prev_bowg_vec = kf_bowgVector;
        
        if (USE_DISTRIBUTION) {
            db_wg.dist_add(in_kernel_vec);
            db_wg.prev_dist_vec = in_kernel_vec;
        }

        if (run_legacy_acceptance) {
        // positive indices, to record the results
        std::vector<int> positive_indices;

        // consider match grouping
        // need to first sort ret in ascending ID
        std::vector<std::vector<int>> groupMatches;
        if (db_wg.matchGrouping(ret, groupMatches, SIMILARITY_TH, MAX_INTRAISLAND_GAP)) {
            // match a island with the highest island score
            int island_ID = db_wg.islandMatching(ret, groupMatches);
            std::vector<int> island = groupMatches[island_ID];

            // fundamental matrix solver
            DVision::FSolver m_fsolver;
            m_fsolver.setImageSize(image.cols, image.rows);

            // temporal consistency check, consider previous k islands 
            if (db_wg.temporalConsistency(ret, island, TEMPORAL_K, MAX_DISTANCE_BETWEEN_GROUPS, MAX_DISTANCE_BETWEEN_QUERIES)) {
                // find the frame that has the largest similarity score
                double max_sim_score = 0;
                BoWG::ItemID loop_ID; // this is the final loop closure candidate
                for (int i = 0; i < island.size(); i++) {
                    if (ret[island[i]].Score > max_sim_score)
                    {
                        loop_ID = ret[island[i]].Id;
                        max_sim_score = ret[island[i]].Score;
                    }
                }

                if (!USE_GEOM) {
                    // if no geometrical checking 
                    positive_indices.push_back(loop_ID);
                    updateRowVector(*cor_row, positive_indices);
                }
                else {
                    // geometric check to finally determine the loop closure frame
                    // Exhaustive searching
                    if (GEOM_DI) {
                        // Direct table searching
                        DBoW2::EntryId best_entry_id = loop_ID;
                        if (isGeometricallyConsistent_DI<DVision::BRIEF::bitset, FBrief>(m_fsolver,db,m_image_descriptors,m_image_keys,best_entry_id,
                                keypoints,brief_descriptors,featvec,MIN_FPOINTS,MAX_REPROJECTION_ERROR,RANSAC_PROBABILITY,MAX_RANSAC_ITERATIONS,MAX_NEIGHBOR_RATIO)){
                            positive_indices.push_back(loop_ID);
                            updateRowVector(*cor_row, positive_indices);
                        }
                    }
                    else {
                        std::vector<cv::DMatch> inlierMatches;
                        if (isGeometricallyConsistent_Exhaustive(map_des[kf_iid],map_kpts[kf_iid],map_des[loop_ID],map_kpts[loop_ID],inlierMatches,MIN_FPOINTS)){
                            positive_indices.push_back(loop_ID);
                            updateRowVector(*cor_row, positive_indices);
                        }
                    }
                }
            }
        }
        }
    }
    else
    {
        if (GEOM_DI)
            db.add(cur_bow_vec, featvec);
        else
            db.add(brief_descriptors);
        
        db_wg.add(kf_bowgVector, kf_iid);

        db_wg.prev_bow_vec = cur_bow_vec;
        db_wg.prev_bowg_vec = kf_bowgVector;
        
        if (USE_DISTRIBUTION) {
            db_wg.dist_add(in_kernel_vec);
            db_wg.prev_dist_vec = in_kernel_vec;
        }
    }

    DetectionResult result;
    result.entry_id = kf_iid;
    result.features.keypoints = keypoints;
    result.features.descriptors = brief_descriptors;
    result.extract_ms = extract_ms;
    result.query_ms = query_ms;
    result.candidates.reserve(ret.size());
    for (const DBoW2::Result& item : ret) {
        Candidate candidate;
        candidate.entry_id = item.Id;
        candidate.score = item.Score;
        result.candidates.push_back(candidate);
    }
    std::sort(
        result.candidates.begin(), result.candidates.end(),
        [](const Candidate& lhs, const Candidate& rhs) {
            if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
            }
            return lhs.entry_id < rhs.entry_id;
        });
    if (result.candidates.size() > static_cast<size_t>(top_k)) {
        result.candidates.resize(static_cast<size_t>(top_k));
    }
    return result;
}

// 函数作用：保存 saveResults 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - filename：输入或输出文件名。
//   - res_matrix：矩阵数据。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGDetector::saveResults(const std::string& filename, const std::vector<std::vector<int>>& res_matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < res_matrix.size(); ++i) {
        bool has_loop = false;
        std::vector<int> loop_frames;

        for (size_t j = 0; j < res_matrix[i].size(); ++j) {
            if (res_matrix[i][j] == 1) {
                has_loop = true;
                loop_frames.push_back(j + 1);
            }
        }

        if (has_loop) {
            file << (i + 1) << ": ";
            for (size_t j = 0; j < loop_frames.size(); ++j) {
                file << loop_frames[j];
                if (j < loop_frames.size() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "Parsed loop closure result saved to " << filename << std::endl;
    std::cout << "---------------------------------------" << std::endl;
}

// 函数作用：执行 initializeGUI 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGDetector::initializeGUI()
{
    cv::namedWindow(main_window, cv::WINDOW_NORMAL);
    cv::resizeWindow(main_window, display_width, display_height);
    display_image = cv::Mat(display_height, display_width, CV_8UC3, cv::Scalar(0, 0, 0));
}

// 函数作用：执行 drawFeatures 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - image：输入图像。
//   - keypoints：图像关键点集合。
//   - wg_keypoints：关键点或关键点集合。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGDetector::drawFeatures(cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::KeyPoint>& wg_keypoints)
{
    cv::Mat overlay = image.clone();
    for (const auto& kp : keypoints) {
        cv::circle(image, kp.pt, kp.size, cv::Scalar(0, 255, 0), 1);
    }
    for (const auto& kp : wg_keypoints) {
        cv::circle(overlay, kp.pt, kp.size, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, kp.pt, kp.size, cv::Scalar(255, 0, 0), 1);
    }

    double alpha = 0.3;
    cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
}

// 函数作用：执行 waitIfPaused 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGDetector::waitIfPaused()
{
    if (!is_paused) {
        if (frame_delay_ms > 0) {
            int steps = frame_delay_ms / 10;
            for (int i = 0; i < steps && !is_paused; i++) {
                char key = cv::waitKey(10);
                if (key == 'p' || key == 'P') {
                    togglePause();
                    break;
                }
            }
            if (!is_paused && frame_delay_ms % 10 > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay_ms % 10));
            }
        }
        return;
    }

    while (is_paused) {
        char key = cv::waitKey(50);
        if (key == 'p' || key == 'P') {
            is_paused = false;
            if (enable_gui) {
                cv::putText(display_image, "Resuming...", 
                           cv::Point(padding, display_height - padding), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                           cv::Scalar(0, 255, 0), 2);
                cv::imshow(main_window, display_image);
                cv::waitKey(1);
            }
            break;
        }
    }
}

// 函数作用：执行 updateGUI 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - current_frame：current_frame 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - keypoints：图像关键点集合。
//   - wg_keypoints：关键点或关键点集合。
//   - loop_frame：loop_frame 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - loop_keypoints：关键点或关键点集合。
//   - loop_wg_keypoints：关键点或关键点集合。
//   - loop_frame_id：节点、词或条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGDetector::updateGUI(const cv::Mat& current_frame, 
                            const std::vector<cv::KeyPoint>& keypoints,
                            const std::vector<cv::KeyPoint>& wg_keypoints,
                            const cv::Mat& loop_frame, 
                            const std::vector<cv::KeyPoint>& loop_keypoints,
                            const std::vector<cv::KeyPoint>& loop_wg_keypoints,
                            int loop_frame_id)
{
    display_image = cv::Mat(display_height, display_width, CV_8UC3, cv::Scalar(240, 240, 240));

    int frame_width = (display_width - 3*padding) / 2;
    int frame_height = display_height - 2*padding - title_height;
    
    cv::rectangle(display_image, 
                 cv::Point(padding, padding + title_height), 
                 cv::Point(padding + frame_width, padding + title_height + frame_height), 
                 cv::Scalar(100, 100, 100), border);
    cv::rectangle(display_image, 
                 cv::Point(2*padding + frame_width, padding + title_height), 
                 cv::Point(2*padding + 2*frame_width, padding + title_height + frame_height), 
                 cv::Scalar(100, 100, 100), border);

    cv::Mat current_display;
    if(!current_frame.empty()) {
        current_display = current_frame.clone();
        drawFeatures(current_display, keypoints, wg_keypoints);
        
        double aspect_ratio = static_cast<double>(current_frame.cols) / current_frame.rows;
        int target_width = frame_width - 2*border;
        int target_height = frame_height - 2*border;
        
        if(static_cast<double>(target_width) / target_height > aspect_ratio) {
            target_width = static_cast<int>(target_height * aspect_ratio);
        } else {
            target_height = static_cast<int>(target_width / aspect_ratio);
        }

        cv::Mat resized_current;
        cv::resize(current_display, resized_current, cv::Size(target_width, target_height));

        int x_offset = padding + border + (frame_width - 2*border - target_width) / 2;
        int y_offset = padding + title_height + border + (frame_height - 2*border - target_height) / 2;

        cv::Mat roi = display_image(cv::Rect(x_offset, y_offset, target_width, target_height));
        resized_current.copyTo(roi);
    }

    std::string current_title = "Current Frame (ID: " + std::to_string(db_wg.cur_image_id) + ")";
    cv::putText(display_image, current_title,
                cv::Point(padding, padding + title_height/2), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    if (!loop_frame.empty()) {
        cv::Mat loop_display = loop_frame.clone();
        drawFeatures(loop_display, loop_keypoints, loop_wg_keypoints);
        
        double aspect_ratio = static_cast<double>(loop_frame.cols) / loop_frame.rows;
        int target_width = frame_width - 2*border;
        int target_height = frame_height - 2*border;
        
        if(static_cast<double>(target_width) / target_height > aspect_ratio) {
            target_width = static_cast<int>(target_height * aspect_ratio);
        } else {
            target_height = static_cast<int>(target_width / aspect_ratio);
        }

        cv::Mat resized_loop;
        cv::resize(loop_display, resized_loop, cv::Size(target_width, target_height));

        int x_offset = 2*padding + frame_width + border + (frame_width - 2*border - target_width) / 2;
        int y_offset = padding + title_height + border + (frame_height - 2*border - target_height) / 2;

        cv::Mat roi = display_image(cv::Rect(x_offset, y_offset, target_width, target_height));
        resized_loop.copyTo(roi);

        std::string loop_title = "Loop Frame (ID: " + std::to_string(loop_frame_id) + ")";
        cv::putText(display_image, loop_title,
                    cv::Point(2*padding + frame_width, padding + title_height/2), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    } else {
        cv::putText(display_image, "No Loop Detected",
                    cv::Point(2*padding + frame_width, padding + title_height/2), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    }

    cv::imshow(main_window, display_image);
    char key = cv::waitKey(1);
    if (key == 'p' || key == 'P') {
        togglePause();
    }
    waitIfPaused();
}

} // namespace BoWGDetector
