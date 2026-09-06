// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: FBrief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions for BRIEF descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

#include "FBrief.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

// 函数作用：执行 meanValue 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - descriptors：BRIEF 描述子集合。
//   - mean：mean 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void FBrief::meanValue(const std::vector<FBrief::pDescriptor> &descriptors, 
  FBrief::TDescriptor &mean)
{
  mean.reset();
  
  if(descriptors.empty()) return;
  
  const int N2 = descriptors.size() / 2;
  const int L = descriptors[0]->size();
  
  vector<int> counters(L, 0);

  vector<FBrief::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FBrief::TDescriptor &desc = **it;
    for(int i = 0; i < L; ++i)
    {
      if(desc[i]) counters[i]++;
    }
  }
  
  for(int i = 0; i < L; ++i)
  {
    if(counters[i] > N2) mean.set(i);
  }
  
}

// --------------------------------------------------------------------------
  
// 函数作用：执行 distance 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - a：a 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - b：b 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double FBrief::distance(const FBrief::TDescriptor &a, 
  const FBrief::TDescriptor &b)
{
  return (double)DVision::BRIEF::distance(a, b);
}

// --------------------------------------------------------------------------
  
// 函数作用：执行 toString 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - a：a 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
std::string FBrief::toString(const FBrief::TDescriptor &a)
{
  // from boost::bitset
  string s;
  to_string(a, s); // reversed
  return s;
}

// --------------------------------------------------------------------------
  
// 函数作用：执行 fromString 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - a：a 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - s：s 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void FBrief::fromString(FBrief::TDescriptor &a, const std::string &s)
{
  // from boost::bitset
  stringstream ss(s);
  ss >> a;
}

// --------------------------------------------------------------------------

// 函数作用：执行 toMat32F 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - descriptors：BRIEF 描述子集合。
//   - mat：输入矩阵。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void FBrief::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  const int L = descriptors[0].size();
  
  mat.create(N, L, CV_32F);
  
  for(int i = 0; i < N; ++i)
  {
    const TDescriptor& desc = descriptors[i];
    float *p = mat.ptr<float>(i);
    for(int j = 0; j < L; ++j, ++p)
    {
      *p = (desc[j] ? 1 : 0);
    }
  } 
}

// --------------------------------------------------------------------------

} // namespace DBoW2

