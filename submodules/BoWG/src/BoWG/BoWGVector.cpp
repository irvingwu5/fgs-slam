// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: BoWGVector.cpp
 * Date: July 2023
 * Author: August 2024
 * Description: bag of word groups vector, used to describe images
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "BoWGVector.h"

namespace BoWG {

// --------------------------------------------------------------------------

// 函数作用：执行 BoWGVector 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BoWGVector::BoWGVector(void)
{
}

// --------------------------------------------------------------------------

// 函数作用：释放对象持有的资源。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BoWGVector::~BoWGVector(void)
{
}

// --------------------------------------------------------------------------

// 函数作用：向数据库添加 addWeight 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - id：条目、词或节点标识符。
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGVector::addWeight(WordGroupID id, WordGroupValue v)
{
    BoWGVector::iterator vit = this->lower_bound(id);

    if(vit != this->end() && !(this->key_comp()(id, vit->first)))
    {
        vit->second += v;
    }
    else
    {
        this->insert(vit, BoWGVector::value_type(id, v));
    }
}

// --------------------------------------------------------------------------

// 函数作用：向数据库添加 addIfNotExist 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - id：条目、词或节点标识符。
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGVector::addIfNotExist(WordGroupID id, WordGroupValue v)
{
    BoWGVector::iterator vit = this->lower_bound(id);

    if(vit == this->end() || (this->key_comp()(id, vit->first)))
    {
        this->insert(vit, BoWGVector::value_type(id, v));
    }
}

// --------------------------------------------------------------------------

// 函数作用：归一化 normalize 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BoWGVector::normalize()
{
    double norm = 0.0;
    BoWGVector::iterator it;

    for(it = begin(); it != end(); ++it)
        norm += it->second * it->second;
    norm = sqrt(norm);

    if(norm > 0.0)
    {
        for(it = begin(); it != end(); ++it)
            it->second /= norm;
    }
}

// --------------------------------------------------------------------------

std::ostream& operator<< (std::ostream &out, const BoWGVector &v)
{
  BoWGVector::const_iterator vit;
  std::vector<unsigned int>::const_iterator iit;
  unsigned int i = 0; 
  const unsigned int N = v.size();
  for(vit = v.begin(); vit != v.end(); ++vit, ++i)
  {
    out << "<" << vit->first << ", " << vit->second << ">";
    
    if(i < N-1) out << ", ";
  }
  return out;
}

} // namespace BoWG