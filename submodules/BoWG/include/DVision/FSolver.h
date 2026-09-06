// 文件作用：提取 BRIEF 特征并验证候选几何一致性；隶属于 BRIEF 特征与几何验证 模块。
/**
 * File: FSolver.h
 * Project: DVision library
 * Author: Dorian Galvez-Lopez
 * Date: November 17, 2011
 * Description: Computes fundamental matrices
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_F_SOLVER__
#define __D_F_SOLVER__

#include <opencv2/core/core.hpp>
#include <vector>

namespace DVision {

/// Computes fundamental matrices
class FSolver
{
public:

  /**
   * Creates the solver without setting the image dimensions
   */
// 函数作用：执行 FSolver 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
  FSolver();
  
  /**
   * Creates the solver and set the image dimensions
   * @param cols width of images
   * @param rows height of images
   */
// 函数作用：执行 FSolver 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - cols：cols 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - rows：rows 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
  FSolver(int cols, int rows);
  
  /**
   * Destructor
   */
// 函数作用：释放对象持有的资源。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
  virtual ~FSolver(){}
  
  /**
   * Sets image size
   * @param cols
   * @param rows
   */
// 函数作用：设置 setImageSize 对应的数据或状态。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - cols：cols 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - rows：rows 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  virtual void setImageSize(int cols, int rows);
  
  /**
   * Finds a fundamental matrix from the given correspondences by running
   * RANSAC
   * @param P1 2xN, 3xN, Nx2, Nx3, correspondences of image 1 in image coordinates
   * @param P2 2xN, 3xN, Nx2, Nx3, correspondences of image 2 in image coordinates
   * @param reprojection_error max reprojection error for getting inliers
   * @param min_points min number of required inliers
   * @param status (out) vector s.t. status[i] == 1 if i-th point 
   *   is an inlier, 0 otherwise
   * @param computeF (default: true) if false, the final F is not computed and
   *   an arbitrary non-empty 3x3 matrix is returned (it saves a svd operation)
   * @param probability RANSAC success probability
   * @param max_its maximum number of RANSAC iterations 
   * @return F s.t. P1' * F * P2 == 0, or empty
   */
// 函数作用：执行 findFundamentalMat 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - P1：P1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - P2：P2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - reprojection_error：reprojection_error 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - min_points：下限参数。
//   - status：status 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - computeF：computeF 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - probability：probability 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_its：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  cv::Mat findFundamentalMat(const cv::Mat &P1, const cv::Mat &P2,
    double reprojection_error, int min_points = 9, 
    std::vector<uchar>* status = NULL,
    bool computeF = true, double probability = 0.99, int max_its = 500) const;

  /**
   * Checks if a consistent fundamental matrix can be computed from the given
   * points. It is not computed, though.
   * @param P1 2xN, 3xN, Nx2, Nx3, correspondences of image 1 in image coordinates
   * @param P2 2xN, 3xN, Nx2, Nx3, correspondences of image 2 in image coordinates
   * @param reprojection_error max reprojection error for getting inliers
   * @param min_points min number of required inliers
   * @param probability RANSAC success probability
   * @param max_its maximum number of RANSAC iterations 
   * @return true iff some fundamental matrix is found
   */
// 函数作用：用基础矩阵 RANSAC 检查几何一致性。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - P1：P1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - P2：P2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - reprojection_error：reprojection_error 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - min_points：下限参数。
//   - probability：probability 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_its：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  bool checkFundamentalMat(const cv::Mat &P1, const cv::Mat &P2,
    double reprojection_error, int min_points = 9,
    double probability = 0.99, int max_its = 500) const;

protected:

  /**
   * Normalize points
   * @param P 2xN, 3xN, Nx2, Nx3, float or double points 
   * @param Q (out) 3xN (double) homogeneous coordinates of points
   */
// 函数作用：归一化 normalizePoints 对应的数据或状态。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - P：P 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - Q：Q 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  void normalizePoints(const cv::Mat &P, cv::Mat &Q) const;
  
  /**
   * Computes F from correspondences Q1(:,i_cols), Q2(:,i_cols)
   * @param Qc1 3xN normalized 
   * @param Qc2 3xN normalized
   * @param i_cols # >= 9
   * @return F12 3x3 or empty
   */
// 函数作用：执行 _computeF 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - Qc1：Qc1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - Qc2：Qc2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - i_cols：i_cols 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  cv::Mat _computeF(const cv::Mat &Qc1, const cv::Mat &Qc2, 
    const std::vector<unsigned int> &i_cols) const;
  
  /**
   * Get inliers in the opencv manner
   * @param Q1 3xN homogeneous points
   * @param Q2 3xN homogeneous points
   * @param F12 fundamental matrix
   * @param err max reprojection error
   * @param status (out) 1xN CV_8U vector with 1 in the positions of the inliers
   */  
  //void getInliers(const cv::Mat &Q1, const cv::Mat &Q2, 
  //  const cv::Mat &F12, double err, cv::Mat &status) const;

protected:
  
  /// Normalization matrix
  cv::Mat m_N;
  
  /// Traspose of normalization matrix
  cv::Mat m_N_t;

};

} // namespace DVision

#endif
