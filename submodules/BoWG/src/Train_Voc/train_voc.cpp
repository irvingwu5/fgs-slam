// 文件作用：提取训练 BRIEF 并生成层次化视觉词典；隶属于 词典训练 模块。
// /**
//  * File: train_bow.cpp
//  * Date: Dec 2024
//  * Author: Xiang Fei
//  * Description: Train a new vocabulary for loop closure
//  */

#include <iostream>
#include "DBoW2.h"
#include <opencv2/features2d.hpp>
#include <time.h>

using namespace std;
using namespace DVision;
using namespace DBoW2;
using namespace cv; 

std::string BRIEF_FILE = "../support_files/brief_pattern.yml";

// 函数作用：训练 trainFeatureExtractor 对应的数据或状态。
// 所属模块：词典训练。
// 输入：
//   - imagePaths：输入图像或图像集合。
//   - vocabularyPath：vocabularyPath 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void trainFeatureExtractor(const vector<string>& imagePaths, const string& vocabularyPath) {
    const int k = 10;
    const int L = 6;
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L1_NORM;

    vector<vector<BRIEF::bitset>> trainingFeatures;
    int imageCounter = 0;

    cout << "[INFO] Starting feature extraction..." << endl;

    for (const auto& path : imagePaths) {
        cout << "[INFO] Processing image: " << path << endl;

        Mat image = imread(path, IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "[ERROR] Failed to load image: " << path << endl;
            continue;
        }

        vector<Point2f> keypoints;
        goodFeaturesToTrack(image, keypoints, 900, 0.001, 10, cv::noArray(), 5);

        if (keypoints.empty()) {
            cerr << "[WARNING] No keypoints detected in image: " << path << endl;
            continue;
        }
        cout << "[INFO] Detected " << keypoints.size() << " keypoints." << endl;

        vector<KeyPoint> cvKeypoints;
        for (const auto& pt : keypoints) {
            cvKeypoints.emplace_back(pt, 5.f);
        }

        vector<BRIEF::bitset> brief_descriptors;
        BriefExtractor extractor(BRIEF_FILE.c_str());
        extractor(image, cvKeypoints, brief_descriptors);

        if (!brief_descriptors.empty()) {
            trainingFeatures.push_back(brief_descriptors);
            cout << "[INFO] Extracted " << brief_descriptors.size() << " descriptors." << endl;
        } else {
            cerr << "[WARNING] No descriptors extracted for image: " << path << endl;
        }

        ++imageCounter;
    }

    cout << "[INFO] Completed feature extraction for " << imageCounter << " images." << endl;

    if (trainingFeatures.empty()) {
        cerr << "[ERROR] No features were extracted. Vocabulary training cannot proceed." << endl;
        return;
    }

    cout << "[INFO] Starting vocabulary training..." << endl;

    BriefVocabulary voc(k, L, weight, scoring);
    voc.create(trainingFeatures);

    cout << "[INFO] Vocabulary training completed successfully." << endl;

    voc.saveBin(vocabularyPath);
    cout << "[INFO] Vocabulary saved to " << vocabularyPath << endl;
}


// 函数作用：解析命令行并运行演示或词典训练入口。
// 所属模块：词典训练。
// 输入：
//   - argc：命令行参数数量。
//   - argv：命令行参数字符串数组。
// 输出：返回进程退出状态码；正常路径生成词典文件并返回 0。
int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <image_folder> <vocabulary_output>" << endl;
        return -1;
    }

    string imageFolder = argv[1];
    string vocabularyPath = argv[2];

    vector<string> imagePaths;
    glob(imageFolder + "/*.png", imagePaths);

    if (imagePaths.empty()) {
        cerr << "No images found in folder: " << imageFolder << endl;
        return -1;
    }

    trainFeatureExtractor(imagePaths, vocabularyPath);

    return 0;
}
