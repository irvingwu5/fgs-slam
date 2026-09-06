// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: ScoringObject.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions to compute bow scores 
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_SCORING_OBJECT__
#define __D_T_SCORING_OBJECT__

#include "BowVector.h"

namespace DBoW2 {

/// Base class of scoring functions
class GeneralScoring
{
public:
  /**
   * Computes the score between two vectors. Vectors must be sorted and 
   * normalized if necessary
   * @param v (in/out)
   * @param w (in/out)
   * @return score
   */
// 函数作用：计算相似度 score 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - w：w 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  virtual double score(const BowVector &v, const BowVector &w) const = 0;

  /**
   * Returns whether a vector must be normalized before scoring according
   * to the scoring scheme
   * @param norm norm to use
   * @return true iff must normalize
   */
// 函数作用：执行 mustNormalize 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - norm：norm 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  virtual bool mustNormalize(LNorm &norm) const = 0;

  /// Log of epsilon
	static const double LOG_EPS; 
  // If you change the type of WordValue, make sure you change also the
	// epsilon value (this is needed by the KL method)
	
// 函数作用：释放对象持有的资源。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
  virtual ~GeneralScoring() {} //!< Required for virtual base classes	
};

/** 
 * Macro for defining Scoring classes
 * @param NAME name of class
 * @param MUSTNORMALIZE if vectors must be normalized to compute the score
 * @param NORM type of norm to use when MUSTNORMALIZE
 */
#define __SCORING_CLASS(NAME, MUSTNORMALIZE, NORM) \
  NAME: public GeneralScoring \
  { public: \
    /** \
     * Computes score between two vectors \
     * @param v \
     * @param w \
     * @return score between v and w \
     */ \
    virtual double score(const BowVector &v, const BowVector &w) const; \
    \
    /** \
     * Says if a vector must be normalized according to the scoring function \
     * @param norm (out) if true, norm to use
     * @return true iff vectors must be normalized \
     */ \
    virtual inline bool mustNormalize(LNorm &norm) const  \
      { norm = NORM; return MUSTNORMALIZE; } \
  }
  
/// L1 Scoring object
// 函数作用：执行 __SCORING_CLASS 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - L1Scoring：L1Scoring 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - true：true 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - L1：L1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
class __SCORING_CLASS(L1Scoring, true, L1);

/// L2 Scoring object
// 函数作用：执行 __SCORING_CLASS 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - L2Scoring：L2Scoring 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - true：true 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - L2：L2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
class __SCORING_CLASS(L2Scoring, true, L2);

/// Chi square Scoring object
// 函数作用：执行 __SCORING_CLASS 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - ChiSquareScoring：ChiSquareScoring 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - true：true 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - L1：L1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
class __SCORING_CLASS(ChiSquareScoring, true, L1);

/// KL divergence Scoring object
// 函数作用：执行 __SCORING_CLASS 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - KLScoring：KLScoring 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - true：true 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - L1：L1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
class __SCORING_CLASS(KLScoring, true, L1);

/// Bhattacharyya Scoring object
// 函数作用：执行 __SCORING_CLASS 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - BhattacharyyaScoring：BhattacharyyaScoring 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - true：true 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - L1：L1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
class __SCORING_CLASS(BhattacharyyaScoring, true, L1);

/// Dot product Scoring object
// 函数作用：执行 __SCORING_CLASS 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - DotProductScoring：DotProductScoring 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - false：false 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - L1：L1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
class __SCORING_CLASS(DotProductScoring, false, L1);

#undef __SCORING_CLASS
  
} // namespace DBoW2

#endif

