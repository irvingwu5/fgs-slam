// 文件作用：提取 BRIEF 特征并验证候选几何一致性；隶属于 BRIEF 特征与几何验证 模块。
/**
 * File: BriefExtractor.cpp
 * Date: Dec 2024
 * Author: Xiang Fei
 * Description: the extractor of brief descriptors
 * Cite: VINS-Mono
 *
 */

#include "BRIEF.h"
#include "BriefExtractor.h"
#include "DUtils.h"
#include <boost/dynamic_bitset.hpp>
#include <vector>

using namespace std;
using namespace DVision;

void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys,
								vector<BRIEF::bitset> &descriptors) const
{
	m_brief.compute(im, keys, descriptors);
}

// 函数作用：执行 BriefExtractor 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - pattern_file：输入或输出文件。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
	// loads the pattern
	cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
	if (!fs.isOpened())
		throw string("Could not open file ") + pattern_file;

	vector<int> x1, y1, x2, y2;
	fs["x1"] >> x1;
	fs["x2"] >> x2;
	fs["y1"] >> y1;
	fs["y2"] >> y2;

	m_brief.importPairs(x1, y1, x2, y2);
}
