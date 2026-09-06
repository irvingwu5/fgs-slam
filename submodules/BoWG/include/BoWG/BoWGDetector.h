// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: BoWGDetector.h
 * Date: August 2024
 * Author: Xiang Fei
 * Description: A loop closure detector using BoWG, an example usage is shown in demo.cpp 
 * When you want to integrate BoWG into other SLAM systems
 * I strongly recommend you to read this file
 *
 */

#ifndef __BOWG_DETECTOR__
#define __BOWG_DETECTOR__

#include <iostream>
#include "BoWGDatabase.h"
#include "DBoW2.h"
#include "GeometicUtils.h"
#include "Parameters.h"
#include <opencv2/features2d.hpp>
#include <chrono>
#include <fstream>
#include <thread>

namespace BoWG {

class BoWGDetector 
{
public:
    /**
    * Constructor of BoWGDetector
    */
// 函数作用：执行 BoWGDetector 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    BoWGDetector(void);

// 函数作用：执行 BoWGDetector 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - use_gui：use_gui 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - delay_ms：delay_ms 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    BoWGDetector(bool use_gui = false, int delay_ms = 0);

    /**
    * Destructor of BoWGDetector
    */
// 函数作用：释放对象持有的资源。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    ~BoWGDetector(void);
    
    // Used to record the result after each query
// 函数作用：执行 updateRowVector 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - row_vec：row_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - indices：indices 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void updateRowVector(std::vector<int>& row_vec, const std::vector<int>& indices);
    
    // process each image, and detect loop closure
// 函数作用：提取图像特征、查询历史数据库并输出回环候选。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - image：输入图像。
//   - is_query：is_query 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - size：size 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - cor_row：cor_row 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void image_detect(cv::Mat image, bool is_query, int size, std::vector<int> &cor_row);

    // Used to generate the final result matrix
// 函数作用：保存 saveResults 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - filename：输入或输出文件名。
//   - res_matrix：矩阵数据。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void saveResults(const std::string& filename, const std::vector<std::vector<int>>& res_matrix);

    // For GUI Visualization
// 函数作用：执行 initializeGUI 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void initializeGUI();
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
    void updateGUI(const cv::Mat& current_frame, 
                  const std::vector<cv::KeyPoint>& keypoints,
                  const std::vector<cv::KeyPoint>& wg_keypoints,
                  const cv::Mat& loop_frame = cv::Mat(), 
                  const std::vector<cv::KeyPoint>& loop_keypoints = std::vector<cv::KeyPoint>(),
                  const std::vector<cv::KeyPoint>& loop_wg_keypoints = std::vector<cv::KeyPoint>(),
                  int loop_frame_id = -1);
// 函数作用：执行 drawFeatures 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - image：输入图像。
//   - keypoints：图像关键点集合。
//   - wg_keypoints：关键点或关键点集合。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void drawFeatures(cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::KeyPoint>& wg_keypoints);

    // Pause/Resume functionality
// 函数作用：执行 togglePause 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void togglePause() { is_paused = !is_paused; }
// 函数作用：执行 isPaused 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    bool isPaused() const { return is_paused; }
// 函数作用：执行 waitIfPaused 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void waitIfPaused();

    BriefDatabase db;
    BriefVocabulary *voc;
    BoWG::BoWGDatabase db_wg;
    std::map<BoWG::ItemID, std::vector<BRIEF::bitset>> map_des;
    std::map<BoWG::ItemID, std::vector<cv::KeyPoint>> map_kpts;
    std::vector<std::vector<BRIEF::bitset>> m_image_descriptors;
    std::vector<std::vector<cv::KeyPoint>> m_image_keys;
    std::vector<std::vector<cv::KeyPoint>> m_image_wg_keys;

    // GUI variables
    cv::Mat display_image;
    const std::string main_window = "Loop Closure Detection";
    const int title_height = 30;
    const int padding = 10;
    const int border = 2;
    const int display_width = 1920;
    const int display_height = 760; // 720 + title_height + padding

    bool enable_gui;
    int frame_delay_ms;  // Delay between frames in milliseconds
    bool is_paused;      // Pause state
};

} // namespace BoWG


#endif