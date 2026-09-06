// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: QueryResults.cpp
 * Date: March, November 2011
 * Author: Dorian Galvez-Lopez
 * Description: structure to store results of database queries
 * License: see the LICENSE.txt file
 *
 */

#include <iostream>
#include <fstream>
#include "QueryResults.h"

using namespace std;

namespace DBoW2
{

// ---------------------------------------------------------------------------

ostream & operator<<(ostream& os, const Result& ret )
{
  os << "<EntryId: " << ret.Id << ", Score: " << ret.Score << ">";
  return os;
}

// ---------------------------------------------------------------------------

ostream & operator<<(ostream& os, const QueryResults& ret )
{
  if(ret.size() == 1)
    os << "1 result:" << endl;
  else
    os << ret.size() << " results:" << endl;
    
  QueryResults::const_iterator rit;
  for(rit = ret.begin(); rit != ret.end(); ++rit)
  {
    os << *rit;
    if(rit + 1 != ret.end()) os << endl;
  }
  return os;
}

// ---------------------------------------------------------------------------

// 函数作用：保存 saveM 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - filename：输入或输出文件名。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void QueryResults::saveM(const std::string &filename) const
{
  fstream f(filename.c_str(), ios::out);
  
  QueryResults::const_iterator qit;
  for(qit = begin(); qit != end(); ++qit)
  {
    f << qit->Id << " " << qit->Score << endl;
  }
  
  f.close();
}

// ---------------------------------------------------------------------------

} // namespace DBoW2

