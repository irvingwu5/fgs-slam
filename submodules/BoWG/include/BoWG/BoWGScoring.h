// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: BoWGScoring.h
 * Date: August 2024
 * Author: Xiang Fei
 * Description: functions to compute bowg scores
 *
 */

#ifndef __BOWG_SCORING__
#define __BOWG_SCORING__

#include "BoWGVector.h"
#include "DBoW2.h"
#include "TemplatedDatabase.h"
#include "TemplatedVocabulary.h"

namespace BoWG {

class BoWGScoring
{
public:
    /**
    * Constructor of BoWGVector
    */
// 函数作用：执行 BoWGScoring 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    BoWGScoring(void);


    /**
    * Destructor of BoWGVector
    */
// 函数作用：释放对象持有的资源。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    ~BoWGScoring(void);


    /**
     * Computes the scores between two vectors.
     * @param v1 (in/out)
     * @param v2 (in/out)
     * @return score
    */
// 函数作用：计算相似度 scoreL1 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreL1(const BoWGVector &v1, const BoWGVector &v2) const;
// 函数作用：计算相似度 scoreL2 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreL2(const BoWGVector &v1, const BoWGVector &v2) const;
// 函数作用：计算相似度 scoreChiSquare 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreChiSquare(const BoWGVector &v1, const BoWGVector &v2) const;
// 函数作用：计算相似度 scoreKL 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreKL(const BoWGVector &v1, const BoWGVector &v2) const;
// 函数作用：计算相似度 scoreBhattacharyya 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreBhattacharyya(const BoWGVector &v1, const BoWGVector &v2) const;
// 函数作用：计算相似度 scoreDot 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreDot(const BoWGVector &v1, const BoWGVector &v2) const;

// 函数作用：计算相似度 scoreL1 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreL1(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
// 函数作用：计算相似度 scoreL2 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreL2(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
// 函数作用：计算相似度 scoreChiSquare 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreChiSquare(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
// 函数作用：计算相似度 scoreKL 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreKL(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
// 函数作用：计算相似度 scoreBhattacharyya 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreBhattacharyya(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
// 函数作用：计算相似度 scoreDot 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v1：v1 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - v2：v2 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double scoreDot(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;

// 函数作用：执行 dis_score 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - w：w 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    double dis_score(const std::vector<int> &v, const std::vector<int> &w) const;

// 函数作用：执行 kernel_normalize 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    std::vector<double> kernel_normalize(const std::vector<int>& v) const;
};


} // namespace BoWG

#endif