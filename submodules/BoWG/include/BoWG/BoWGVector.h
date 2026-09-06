// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: BoWGVector.h
 * Date: August 2024
 * Author: Xiang Fei
 * Description: bag of word groups vector, used to describe images
*/

#ifndef __BOWG_VECTOR__
#define __BOWG_VECTOR__

#include <iostream>
#include <map>
#include <vector>

namespace BoWG {
    
// ID of Word Groups
typedef unsigned int WordGroupID;

// Value of a Word Group
typedef double WordGroupValue;

// Vector of word groups to represent images
class BoWGVector:
    public std::map<WordGroupID, WordGroupValue>
{
public:

    /**
    * Constructor of BoWGVector
    */
// 函数作用：执行 BoWGVector 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    BoWGVector(void);

    /**
    * Destructor of BoWGVector
    */
// 函数作用：释放对象持有的资源。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    ~BoWGVector(void);

    /**
    * Add a value to a word groups value existing in the vector, or create a new 
    * word group with the given value, used for TF-IDF.
    * @param id word group id to look for
    * @param v value to create the word group with, or to add to existing word group
    */
// 函数作用：向数据库添加 addWeight 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - id：条目、词或节点标识符。
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void addWeight(WordGroupID id, WordGroupValue v);

    /**
     * Adds a word group with a value to the vector only if this does not exist yet
     * used for IDF or BINARY
     * @param id word group id to look for
     * @param v value to give to the word group if this does not exist
    */
// 函数作用：向数据库添加 addIfNotExist 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - id：条目、词或节点标识符。
//   - v：v 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void addIfNotExist(WordGroupID id, WordGroupValue v);

    /**
     * L-Normalizes the values in the vector
    */
// 函数作用：归一化 normalize 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void normalize();

    /**
     * Prints the content of the bowg vector
     * @param out stream'
     * @param v
    */
    friend std::ostream& operator<<(std::ostream &out, const BoWGVector &v);
};

} // namespace BoWG

#endif