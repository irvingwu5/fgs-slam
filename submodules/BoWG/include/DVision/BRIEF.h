// 文件作用：提取 BRIEF 特征并验证候选几何一致性；隶属于 BRIEF 特征与几何验证 模块。
/**
 * File: BRIEF.h
 * Author: Dorian Galvez-Lopez
 * Date: March 2011
 * Description: implementation of BRIEF (Binary Robust Independent 
 *   Elementary Features) descriptor by 
 *   Michael Calonder, Vincent Lepetit and Pascal Fua
 *   + close binary tests (by Dorian Galvez-Lopez)
 *
 * If you use this code with the RANDOM_CLOSE descriptor version, please cite:
  @INPROCEEDINGS{GalvezIROS11,
    author={Galvez-Lopez, Dorian and Tardos, Juan D.},
    booktitle={Intelligent Robots and Systems (IROS), 2011 IEEE/RSJ International Conference on},
    title={Real-time loop detection with bags of binary words},
    year={2011},
    month={sept.},
    volume={},
    number={},
    pages={51 -58},
    keywords={},
    doi={10.1109/IROS.2011.6094885},
    ISSN={2153-0858}
  }
 *
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_BRIEF__
#define __D_BRIEF__

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <boost/dynamic_bitset.hpp>

namespace DVision {

/// BRIEF descriptor
class BRIEF
{
public:

  /// Bitset type
  typedef boost::dynamic_bitset<> bitset;

  /// Type of pairs
  enum Type
  {
    RANDOM, // random pairs (Calonder's original version)
    RANDOM_CLOSE, // random but close pairs (used in GalvezIROS11)
  };
  
public:

  /**
   * Creates the BRIEF a priori data for descriptors of nbits length
   * @param nbits descriptor length in bits
   * @param patch_size 
   * @param type type of pairs to generate
   */
// 函数作用：执行 BRIEF 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - nbits：nbits 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - patch_size：patch_size 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - type：type 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
  BRIEF(int nbits = 256, int patch_size = 48, Type type = RANDOM_CLOSE);
// 函数作用：释放对象持有的资源。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
  virtual ~BRIEF();
  
  /**
   * Returns the descriptor length in bits
   * @return descriptor length in bits
   */
// 函数作用：获取 getDescriptorLengthInBits 对应的数据或状态。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  inline int getDescriptorLengthInBits() const
  {
    return m_bit_length;
  }
  
  /**
   * Returns the type of classifier
   */
// 函数作用：获取 getType 对应的数据或状态。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  inline Type getType() const
  {
    return m_type;
  }
  
  /**
   * Returns the size of the patch
   */
// 函数作用：获取 getPatchSize 对应的数据或状态。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  inline int getPatchSize() const
  {
    return m_patch_size;
  }
  
  /**
   * Returns the BRIEF descriptors of the given keypoints in the given image
   * @param image
   * @param points
   * @param descriptors 
   * @param treat_image (default: true) if true, the image is converted to 
   *   grayscale if needed and smoothed. If not, it is assumed the image has
   *   been treated by the user
   * @note this function is similar to BRIEF::compute
   */
  inline void operator() (const cv::Mat &image, 
    const std::vector<cv::KeyPoint> &points,
    std::vector<bitset> &descriptors,
    bool treat_image = true) const
  {
    compute(image, points, descriptors, treat_image);
  }
  
  /**
   * Returns the BRIEF descriptors of the given keypoints in the given image
   * @param image
   * @param points
   * @param descriptors 
   * @param treat_image (default: true) if true, the image is converted to 
   *   grayscale if needed and smoothed. If not, it is assumed the image has
   *   been treated by the user
   * @note this function is similar to BRIEF::operator()
   */ 
// 函数作用：在关键点位置计算 BRIEF 二进制描述子。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - image：输入图像。
//   - points：points 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - descriptors：BRIEF 描述子集合。
//   - treat_image：输入图像或图像集合。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  void compute(const cv::Mat &image,
    const std::vector<cv::KeyPoint> &points,
    std::vector<bitset> &descriptors,
    bool treat_image = true) const;
  
  /**
   * Exports the test pattern
   * @param x1 x1 coordinates of pairs
   * @param y1 y1 coordinates of pairs
   * @param x2 x2 coordinates of pairs
   * @param y2 y2 coordinates of pairs
   */
// 函数作用：执行 exportPairs 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - x1：x1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - y1：y1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - x2：x2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - y2：y2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  inline void exportPairs(std::vector<int> &x1, std::vector<int> &y1,
    std::vector<int> &x2, std::vector<int> &y2) const
  {
    x1 = m_x1;
    y1 = m_y1;
    x2 = m_x2;
    y2 = m_y2;
  }
  
  /**
   * Sets the test pattern
   * @param x1 x1 coordinates of pairs
   * @param y1 y1 coordinates of pairs
   * @param x2 x2 coordinates of pairs
   * @param y2 y2 coordinates of pairs
   */
// 函数作用：执行 importPairs 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - x1：x1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - y1：y1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - x2：x2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - y2：y2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  inline void importPairs(const std::vector<int> &x1, 
    const std::vector<int> &y1, const std::vector<int> &x2, 
    const std::vector<int> &y2)
  {
    m_x1 = x1;
    m_y1 = y1;
    m_x2 = x2;
    m_y2 = y2;
    m_bit_length = x1.size();
  }
  
  /**
   * Returns the Hamming distance between two descriptors
   * @param a first descriptor vector
   * @param b second descriptor vector
   * @return hamming distance
   */
// 函数作用：执行 distance 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - a：a 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - b：b 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  inline static int distance(const bitset &a, const bitset &b)
  {
    return (a^b).count();
  }

protected:

  /**
   * Generates random points in the patch coordinates, according to 
   * m_patch_size and m_bit_length
   */
// 函数作用：执行 generateTestPoints 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  void generateTestPoints();
  
protected:

  /// Descriptor length in bits
  int m_bit_length;

  /// Patch size
  int m_patch_size;
  
  /// Type of pairs
  Type m_type;

  /// Coordinates of test points relative to the center of the patch
  std::vector<int> m_x1, m_x2;
  std::vector<int> m_y1, m_y2;

};

} // namespace DVision

#endif