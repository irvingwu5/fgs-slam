// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: FClass.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: generic FClass to instantiate templated classes
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_FCLASS__
#define __D_T_FCLASS__

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace DBoW2 {

/// Generic class to encapsulate functions to manage descriptors.
/**
 * This class must be inherited. Derived classes can be used as the
 * parameter F when creating Templated structures
 * (TemplatedVocabulary, TemplatedDatabase, ...)
 */
class FClass
{
  class TDescriptor;
  typedef const TDescriptor *pDescriptor;
  
  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
// 函数作用：执行 meanValue 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - descriptors：BRIEF 描述子集合。
//   - mean：mean 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  virtual void meanValue(const std::vector<pDescriptor> &descriptors, 
    TDescriptor &mean) = 0;
  
  /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
// 函数作用：执行 distance 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - a：a 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - b：b 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  static double distance(const TDescriptor &a, const TDescriptor &b);
  
  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
// 函数作用：执行 toString 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - a：a 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  static std::string toString(const TDescriptor &a);
  
  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
// 函数作用：执行 fromString 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - a：a 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - s：s 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  static void fromString(TDescriptor &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
// 函数作用：执行 toMat32F 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - descriptors：BRIEF 描述子集合。
//   - mat：输入矩阵。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  static void toMat32F(const std::vector<TDescriptor> &descriptors, 
    cv::Mat &mat);
};

} // namespace DBoW2

#endif
