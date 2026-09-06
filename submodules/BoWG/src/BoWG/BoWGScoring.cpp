// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: BoWGScoring.cpp
 * Date: July 2023
 * Author: August 2024
 * Description: functions to compute bowg scores
 *
 */
#include <cfloat>
#include <cmath>
#include "BoWGVector.h"
#include "BoWGScoring.h"

namespace BoWG{

// 函数作用：执行 BoWGScoring 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BoWGScoring::BoWGScoring(void)
{
}

// --------------------------------------------------------------------------

// 函数作用：释放对象持有的资源。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BoWGScoring::~BoWGScoring(void)
{
}

// 函数作用：计算相似度 scoreL1 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreL1(const BoWGVector &v1, const BoWGVector &v2) const
{
  BoWGVector::const_iterator v1_it, v2_it;
  const BoWGVector::const_iterator v1_end = v1.end();
  const BoWGVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordGroupValue& vi = v1_it->second;
    const WordGroupValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += fabs(vi - wi) - fabs(vi) - fabs(wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }
  
  // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|) 
  //		for all i | v_i != 0 and w_i != 0 
  // (Nister, 2006)
  // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
  score = -score/2.0;

  return score; // [0..1]
}


// 函数作用：计算相似度 scoreL2 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreL2(const BoWGVector &v1, const BoWGVector &v2) const
{
  BoWGVector::const_iterator v1_it, v2_it;
  const BoWGVector::const_iterator v1_end = v1.end();
  const BoWGVector::const_iterator v2_end = v2.end();

  v1_it = v1.begin();
  v2_it = v2.begin();

  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
      const WordGroupValue& vi = v1_it->second;
      const WordGroupValue& wi = v2_it->second;

      if(v1_it->first == v2_it->first)
      {
          // std::cout << "vi:" << vi << std::endl;
          // std::cout << "wi:" << wi << std::endl;
          score += vi * wi;

          // move v1 and v2 forward
          ++v1_it;
          ++v2_it;
      }
      else if(v1_it->first < v2_it->first)
      {
          // move v1 forward
          v1_it = v1.lower_bound(v2_it->first);
      }
      else{
          // move v2 forward
          v2_it = v2.lower_bound(v1_it->first);
      }
  }
  // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
//		for all i | v_i != 0 and w_i != 0 )
// (Nister, 2006)
  if(score >= 1) // rounding errors
      score = 1.0;
  else
      score = 1.0 - sqrt(1.0 - score); // [0..1]

  return score;
}


// 函数作用：计算相似度 scoreChiSquare 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreChiSquare(const BoWGVector &v1, const BoWGVector &v2) 
  const
{
  BoWGVector::const_iterator v1_it, v2_it;
  const BoWGVector::const_iterator v1_end = v1.end();
  const BoWGVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  // all the items are taken into account
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordGroupValue& vi = v1_it->second;
    const WordGroupValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      // (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
      // we move the -4 out
      if(vi + wi != 0.0) score += vi * wi / (vi + wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
    }
  }
    
  // this takes the -4 into account
  score = 2. * score; // [0..1]

  return score;
}

// 函数作用：计算相似度 scoreKL 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreKL(const BoWGVector &v1, const BoWGVector &v2) const
{ 
  BoWGVector::const_iterator v1_it, v2_it;
  const BoWGVector::const_iterator v1_end = v1.end();
  const BoWGVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  // all the items or v are taken into account
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordGroupValue& vi = v1_it->second;
    const WordGroupValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      if(vi != 0 && wi != 0) score += vi * log(vi/wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      score += vi * (log(vi) - DBoW2::GeneralScoring::LOG_EPS);
      ++v1_it;
    }
    else
    {
      // move v2_it forward, do not add any score
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }
  
  // sum rest of items of v
  for(; v1_it != v1_end; ++v1_it) 
    if(v1_it->second != 0)
      score += v1_it->second * (log(v1_it->second) - DBoW2::GeneralScoring::LOG_EPS);
  
  return score; // cannot be scaled
}


// 函数作用：计算相似度 scoreBhattacharyya 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreBhattacharyya(const BoWGVector &v1, 
  const BoWGVector &v2) const
{
  BoWGVector::const_iterator v1_it, v2_it;
  const BoWGVector::const_iterator v1_end = v1.end();
  const BoWGVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordGroupValue& vi = v1_it->second;
    const WordGroupValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += sqrt(vi * wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  return score; // already scaled
}


// 函数作用：计算相似度 scoreDot 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreDot(const BoWGVector &v1, 
  const BoWGVector &v2) const
{
  BoWGVector::const_iterator v1_it, v2_it;
  const BoWGVector::const_iterator v1_end = v1.end();
  const BoWGVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordGroupValue& vi = v1_it->second;
    const WordGroupValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += vi * wi;
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  return score; // cannot scale
}


// 函数作用：计算相似度 scoreL1 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreL1(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const
{
  DBoW2::BowVector::const_iterator v1_it, v2_it;
  const DBoW2::BowVector::const_iterator v1_end = v1.end();
  const DBoW2::BowVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const DBoW2::WordValue& vi = v1_it->second;
    const DBoW2::WordValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += fabs(vi - wi) - fabs(vi) - fabs(wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }
  
  // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|) 
  //		for all i | v_i != 0 and w_i != 0 
  // (Nister, 2006)
  // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
  score = -score/2.0;

  return score; // [0..1]
}


// 函数作用：计算相似度 scoreL2 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreL2(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const
{
  DBoW2::BowVector::const_iterator v1_it, v2_it;
  const DBoW2::BowVector::const_iterator v1_end = v1.end();
  const DBoW2::BowVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const DBoW2::WordValue& vi = v1_it->second;
    const DBoW2::WordValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += vi * wi;
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }
  
  // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
	//		for all i | v_i != 0 and w_i != 0 )
	// (Nister, 2006)
	if(score >= 1) // rounding errors
	  score = 1.0;
	else
    score = 1.0 - sqrt(1.0 - score); // [0..1]

  return score;
}


// 函数作用：计算相似度 scoreChiSquare 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreChiSquare(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) 
  const
{
  DBoW2::BowVector::const_iterator v1_it, v2_it;
  const DBoW2::BowVector::const_iterator v1_end = v1.end();
  const DBoW2::BowVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  // all the items are taken into account
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const DBoW2::WordValue& vi = v1_it->second;
    const DBoW2::WordValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      // (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
      // we move the -4 out
      if(vi + wi != 0.0) score += vi * wi / (vi + wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
    }
  }
    
  // this takes the -4 into account
  score = 2. * score; // [0..1]

  return score;
}


// 函数作用：计算相似度 scoreKL 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreKL(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const
{ 
  DBoW2::BowVector::const_iterator v1_it, v2_it;
  const DBoW2::BowVector::const_iterator v1_end = v1.end();
  const DBoW2::BowVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  // all the items or v are taken into account
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const DBoW2::WordValue& vi = v1_it->second;
    const DBoW2::WordValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      if(vi != 0 && wi != 0) score += vi * log(vi/wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      score += vi * (log(vi) - DBoW2::GeneralScoring::LOG_EPS);
      ++v1_it;
    }
    else
    {
      // move v2_it forward, do not add any score
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }
  
  // sum rest of items of v
  for(; v1_it != v1_end; ++v1_it) 
    if(v1_it->second != 0)
      score += v1_it->second * (log(v1_it->second) - DBoW2::GeneralScoring::LOG_EPS);
  
  return score; // cannot be scaled
}


// 函数作用：计算相似度 scoreBhattacharyya 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreBhattacharyya(const DBoW2::BowVector &v1, 
  const DBoW2::BowVector &v2) const
{
  DBoW2::BowVector::const_iterator v1_it, v2_it;
  const DBoW2::BowVector::const_iterator v1_end = v1.end();
  const DBoW2::BowVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const DBoW2::WordValue& vi = v1_it->second;
    const DBoW2::WordValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += sqrt(vi * wi);
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  return score; // already scaled
}


// 函数作用：计算相似度 scoreDot 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::scoreDot(const DBoW2::BowVector &v1, 
  const DBoW2::BowVector &v2) const
{
  DBoW2::BowVector::const_iterator v1_it, v2_it;
  const DBoW2::BowVector::const_iterator v1_end = v1.end();
  const DBoW2::BowVector::const_iterator v2_end = v2.end();
  
  v1_it = v1.begin();
  v2_it = v2.begin();
  
  double score = 0;
  
  while(v1_it != v1_end && v2_it != v2_end)
  {
    const DBoW2::WordValue& vi = v1_it->second;
    const DBoW2::WordValue& wi = v2_it->second;
    
    if(v1_it->first == v2_it->first)
    {
      score += vi * wi;
      
      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  return score; // cannot scale
}


// 函数作用：执行 kernel_normalize 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
std::vector<double> BoWGScoring::kernel_normalize(const std::vector<int>& v) const
{
    double norm = 0.0;
    std::vector<double> v_normalized(v.size(),0);
    for (int i=0; i<v.size(); i++)
        norm += v[i];
    norm = sqrt(norm);
    if(norm > 0.0)
    {
      for(int i=0; i<v.size(); i++)
        v_normalized[i] = v[i] / norm;
    }
    return v_normalized;
}


// 函数作用：执行 dis_score 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - w：w 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double BoWGScoring::dis_score(const std::vector<int> &v, const std::vector<int> &w) const
{
  std::vector<int> v_normalized = v;
  std::vector<int> w_normalized = w;
  int size = w_normalized.size();
  double max_score = 0.0;
  std::vector<int> vec = w_normalized;
  for (int i=0;i<size;i++)
  {
    int first_element = vec.front();
    vec.erase(vec.begin());
    vec.push_back(first_element);
    double dist = 0;
    for (int j=0;j<size;j++)
    {
      dist += sqrt((v_normalized[j]-vec[j])*(v_normalized[j]-vec[j]));
    }
    double score = 1/(1+dist);
    if (max_score<score)
      max_score = score;
  }
  return max_score;
}


} // namespace BoWG