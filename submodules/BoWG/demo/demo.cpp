// 文件作用：驱动图像序列检测、结果保存和检索指标统计；隶属于 演示驱动与评估 模块。
/**
 * File: demo.cpp
 * Date: Dec 2024
 * Author: Xiang Fei
 * Description: demo of BoWG to detect loop closure
 */


#include <iostream>
#include "BoWGDatabase.h"
#include "DBoW2.h"
#include "BoWGDetector.h"
#include <opencv2/features2d.hpp>
#include <chrono>
#include <time.h>
#include <fstream>
#include <string.h>
#include <map>
#include <sstream>

using namespace std;
using namespace BoWG;
using namespace DVision;
using namespace DBoW2;

// 函数作用：统计回环检测的精确率与召回率。
// 所属模块：演示驱动与评估。
// 输入：
//   - gt_file：输入或输出文件。
//   - result_file：结果输出容器。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void calculatePrecisionRecall(const std::string& gt_file, const std::string& result_file) {
    std::map<int, std::vector<int>> gt_dict;
    int num_loop_events = 0;
    std::ifstream gt_stream(gt_file);
    std::string line;
    
    while (std::getline(gt_stream, line)) {
        std::istringstream iss(line);
        std::string key_str, values_str;
        
        std::getline(iss, key_str, ':');
        std::getline(iss, values_str);
        
        int key = std::stoi(key_str);
        
        std::vector<int> values;
        std::istringstream values_stream(values_str);
        std::string value;
        while (std::getline(values_stream, value, ',')) {
            values.push_back(std::stoi(value));
        }
        
        gt_dict[key] = values;
        num_loop_events++;
    }

    int true_positives = 0;
    int false_positives = 0;
    int num_wrong_matching = 0;

    std::ifstream result_stream(result_file);
    while (std::getline(result_stream, line)) {
        std::istringstream iss(line);
        std::string key_str, value_str;
        
        std::getline(iss, key_str, ':');
        std::getline(iss, value_str);
        
        int key = std::stoi(key_str);
        std::vector<int> detected_values;
        std::istringstream values_stream(value_str);
        std::string value;
        while (std::getline(values_stream, value, ',')) {
            detected_values.push_back(std::stoi(value));
        }

        auto gt_it = gt_dict.find(key);
        if (gt_it == gt_dict.end()) {
            std::cout << "false positive at: " << key << std::endl;
            false_positives++;
            continue;
        }

        bool found_match = false;
        for (int detected_value : detected_values) {
            if (std::find(gt_it->second.begin(), gt_it->second.end(), detected_value) != gt_it->second.end()) {
                true_positives++;
                found_match = true;
            }
        }
        
        if (!found_match) {
            std::cout << "wrong matching at: " << key << std::endl;
            num_wrong_matching++;
        }
    }

    std::cout << "Loop Events Num: " << num_loop_events << std::endl;
    std::cout << "True Positives: " << true_positives << std::endl;
    std::cout << "False Positives: " << false_positives << std::endl;
    std::cout << "Wrong Matching Num: " << num_wrong_matching << std::endl;

    std::cout << "--------------------Final Results---------------------" << std::endl;
    double precision = (static_cast<double>(true_positives) / 
                       (true_positives + false_positives + num_wrong_matching)) * 100;
    double recall = (static_cast<double>(true_positives) / num_loop_events) * 100;
    
    std::cout << "Precision: " << std::fixed << std::setprecision(2) << precision << "%" << std::endl;
    std::cout << "Recall: " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
}


// 函数作用：打印 printHelp 对应的数据或状态。
// 所属模块：演示驱动与评估。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void printHelp() {
    cout << "Usage: ./demo [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --gui              Enable GUI visualization" << endl;
    cout << "  --delay <ms>       Set delay between frames in milliseconds, for better visualization" << endl;
    cout << "  --config <path>    Path to configuration file (default: ../config/demo.yaml)" << endl;
    cout << "  --help             Show this help message" << endl;
    cout << endl;
    cout << "Example:" << endl;
    cout << "  ./demo --config ../config/my_config.yaml --gui --delay 20" << endl;
    cout << endl;
    cout << "Controls:" << endl;
    cout << "  Press 'P' to pause/resume" << endl;
}

// 函数作用：解析命令行并运行演示或词典训练入口。
// 所属模块：演示驱动与评估。
// 输入：
//   - argc：命令行参数数量。
//   - argv：命令行参数字符串数组。
// 输出：返回进程退出状态码；0 表示正常结束，非零值表示参数或运行错误。
int main(int argc, char** argv)
{

    bool use_gui = false;
    int delay_ms = 0;
    string config_path = "../config/demo_NC.yaml";  // default config path

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gui") == 0) {
            use_gui = true;
        } 
        else if (strcmp(argv[i], "--delay") == 0) {
            if (i + 1 < argc) {
                delay_ms = atoi(argv[++i]);
            } else {
                cerr << "Error: --delay requires a milliseconds argument" << endl;
                return -1;
            }
        }
        else if (strcmp(argv[i], "--config") == 0) {
            if (i + 1 < argc) {
                config_path = argv[++i];
            } else {
                cerr << "Error: --config requires a path argument" << endl;
                return -1;
            }
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printHelp();
            return 0;
        }
    }

    cout << "*** BoWG Loop Closure Test ***" << endl;
    if (!Parameters::readParameters(config_path)) {
        std::cerr << "Failed to read parameters from: " << config_path << std::endl;
        return -1;
    }

    vector<cv::String> imagePaths;
    cv::glob(IMAGE_PATH, imagePaths, false);

    auto start = std::chrono::high_resolution_clock::now();

    BoWG::BoWGDetector bowg_detector(use_gui, delay_ms);

    std::vector<cv::Mat> all_images;
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<std::vector<cv::KeyPoint>> all_wg_keypoints;

    // correspondence matrix
    std::vector<std::vector<int>> cor_matrix(DISLOCAL, std::vector<int>(imagePaths.size(), 0));

    for (int i = 0; i < imagePaths.size(); i++)
    {
        cout << "---------------------------------------" << endl;
        cout << "current image id: " << bowg_detector.db_wg.cur_image_id << endl;
        cv::Mat image;
        image = cv::imread(imagePaths[i]);
        if(image.data == nullptr)
        {
            cout<<"image doen't exist"<<endl;
            return 0;
        }

        if (use_gui) {
            all_images.push_back(image.clone());
        }

        if (i<DISLOCAL) {
            std::vector<int> cor_row(imagePaths.size(), 0);
            bowg_detector.image_detect(image,false,i+1,cor_row);
            if (use_gui) {
                all_keypoints.push_back(bowg_detector.m_image_keys.back());
                all_wg_keypoints.push_back(bowg_detector.m_image_wg_keys.back());
                bowg_detector.updateGUI(image, bowg_detector.m_image_keys.back(), bowg_detector.m_image_wg_keys.back());
            }
        }
        else {
            std::vector<int> cor_row(imagePaths.size(), 0);
            bowg_detector.image_detect(image,true,i+1,cor_row);
            cor_matrix.push_back(cor_row);
            if (use_gui) {
                all_keypoints.push_back(bowg_detector.m_image_keys.back());
                all_wg_keypoints.push_back(bowg_detector.m_image_wg_keys.back());
                int loop_frame_idx = -1;
                for (int j = 0; j < cor_row.size(); j++) {
                    if (cor_row[j] == 1) {
                        loop_frame_idx = j;
                        break;
                    }
                }
                if (loop_frame_idx != -1) {
                    cout << "Loop detected: Current frame " << i << " matches with frame " << loop_frame_idx << endl;
                    bowg_detector.updateGUI(image, 
                              bowg_detector.m_image_keys.back(),
                              bowg_detector.m_image_wg_keys.back(),
                              all_images[loop_frame_idx], 
                              all_keypoints[loop_frame_idx],
                              all_wg_keypoints[loop_frame_idx],
                              loop_frame_idx);
                } else {
                    bowg_detector.updateGUI(image, bowg_detector.m_image_keys.back(), bowg_detector.m_image_wg_keys.back());
                }
            }
        }
        bowg_detector.db_wg.cur_image_id = bowg_detector.db_wg.cur_image_id + 1;
    }
    
    bowg_detector.saveResults(RESULTS_PATH, cor_matrix);

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9;
    if (!use_gui)
        cout << "time = " << time << "s" << endl;

    if (GT_USE)
        calculatePrecisionRecall(GT_PATH, RESULTS_PATH);

    if (use_gui) {
        cout << "Press any key to exit..." << endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}
