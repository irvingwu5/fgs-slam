// 文件作用：提取 BRIEF 特征并验证候选几何一致性；隶属于 BRIEF 特征与几何验证 模块。
/**
 * File: BriefExtractor.h
 * Date: Dec 2024
 * Author: Xiang Fei
 * Description: the extractor of brief descriptors
 * Cite: VINS-Mono
 *
 */

#ifndef __D_BRIEF_EXTRACTOR__
#define __D_BRIEF_EXTRACTOR__

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <boost/dynamic_bitset.hpp>

namespace DVision {
/**
 * @brief For computing BRIEF descriptors of keypoints using
 * predefined BRIEF patterns
 */
class BriefExtractor
{
public:
	/**
	 * @brief Compute the BRIEF descriptors of the given keypoints in an image
	 *
	 * @param im Input
	 * @param keys Input
	 * @param descriptors Output
	 */
	virtual void operator()(const cv::Mat &im, std::vector<cv::KeyPoint> &keys, std::vector<BRIEF::bitset> &descriptors) const;

	/**
	 * @brief Load the pattern that was used to build the BRIEF vocabulary,
	 * to make the descriptors compatible with the predefined vocabulary.
	 *
	 * This is because DVision::BRIEF computes a random pattern by default
	 * when the object is created.
	 *
	 * @param pattern_file BRIEF pattern file
	 */
// 函数作用：执行 BriefExtractor 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - pattern_file：输入或输出文件。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
	BriefExtractor(const std::string &pattern_file);

	// BRIEF descriptor
	DVision::BRIEF m_brief;
};

} // namespace DVision

#endif