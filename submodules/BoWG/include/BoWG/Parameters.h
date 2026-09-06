// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: Parameters.h
 * Date: Dec 2024
 * Author: Xiang Fei
 * Description: parameters used by BoWG
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <iostream>

extern std::string IMAGE_PATH;
extern std::string RESULTS_PATH;
extern std::string BRIEF_PATTERN_FILE;
extern std::string VOCABULARY_PATH;
extern bool GT_USE;
extern std::string GT_PATH;

extern int DISLOCAL;
extern int FEATURE_CNT;
extern bool MS_FEATURE;
extern int MS_LEVELS;
extern float FEATURE_SIZE;
extern int W_WEIGHT_TYPE;
extern int W_SCORING_TYPE;
extern int WG_WEIGHT_TYPE;
extern int WG_SCORING_TYPE;
extern bool USE_WG;
extern bool USE_DISTRIBUTION;
extern int DISTRIBUTION_BATCH;
extern double W_WEIGHT;
extern double WG_WEIGHT;
extern bool USE_TEMPORAL_SCORE;
extern double PREV_WEIGHT_TH;
extern double TEMPORAL_PARAM;
extern double SIMILARITY_TH;
extern int TEMPORAL_K;
extern int MAX_INTRAISLAND_GAP;
extern int MAX_DISTANCE_BETWEEN_GROUPS;
extern int MAX_DISTANCE_BETWEEN_QUERIES;
extern bool USE_GEOM;
extern bool GEOM_DI;
extern int DI_LEVEL;
extern int MIN_FPOINTS;
extern double MAX_REPROJECTION_ERROR;
extern double RANSAC_PROBABILITY;
extern int MAX_RANSAC_ITERATIONS;
extern double MAX_NEIGHBOR_RATIO;
extern double MIN_PREV_W_SCORE;
extern double MIN_PREV_WG_SCORE;
extern double MIN_PREV_DIST_SCORE;


class Parameters {
public:
// 函数作用：读取 readParameters 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - config_file：输入或输出文件。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    static bool readParameters(const std::string& config_file);
    
    template <typename T>
// 函数作用：读取 readValue 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - fs：fs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - name：name 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - value：value 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - default_value：default_value 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    static bool readValue(const cv::FileStorage& fs, const std::string& name, T& value, const T& default_value = T()) {
        if (fs[name].empty()) {
            std::cerr << name << " not found, using default value: " << default_value << std::endl;
            value = default_value;
            return false;
        }
        fs[name] >> value;
        std::cout << "Loaded " << name << ": " << value << std::endl;
        return true;
    }

// 函数作用：读取 readValue 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - fs：fs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - name：name 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - value：value 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - default_value：default_value 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    static bool readValue(const cv::FileStorage& fs, const std::string& name, 
                         std::string& value, const std::string& default_value = std::string()) {
        if (fs[name].empty()) {
            std::cerr << name << " not found, using default value: " << default_value << std::endl;
            value = default_value;
            return false;
        }
        fs[name] >> value;
        std::cout << "Loaded " << name << ": " << value << std::endl;
        return true;
    }

// 函数作用：读取 readValue 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - fs：fs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - name：name 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - value：value 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - default_value：default_value 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    static bool readValue(const cv::FileStorage& fs, const std::string& name, 
                         std::string& value, const char* default_value) {
        return readValue(fs, name, value, std::string(default_value));
    }
};

#endif /* PARAMETERS_H */
