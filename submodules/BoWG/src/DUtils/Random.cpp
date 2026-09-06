// 文件作用：提供随机数、时间戳和异常等通用能力；隶属于 基础工具 模块。
/*	
 * File: Random.cpp
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010
 * Description: manages pseudo-random numbers
 * License: see the LICENSE.txt file
 *
 */

#include "Random.h"
#include "Timestamp.h"
#include <cstdlib>
using namespace std;

bool DUtils::Random::m_already_seeded = false;

// 函数作用：执行 SeedRand 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void DUtils::Random::SeedRand(){
	Timestamp time;
	time.setToCurrentTime();
	srand((unsigned)time.getFloatTime()); 
}

// 函数作用：执行 SeedRandOnce 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void DUtils::Random::SeedRandOnce()
{
  if(!m_already_seeded)
  {
    DUtils::Random::SeedRand();
    m_already_seeded = true;
  }
}

// 函数作用：执行 SeedRand 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - seed：随机数种子。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void DUtils::Random::SeedRand(int seed)
{
	srand(seed); 
}

// 函数作用：执行 SeedRandOnce 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - seed：随机数种子。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void DUtils::Random::SeedRandOnce(int seed)
{
  if(!m_already_seeded)
  {
    DUtils::Random::SeedRand(seed);
    m_already_seeded = true;
  }
}

// 函数作用：执行 RandomInt 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - min：下限参数。
//   - max：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
int DUtils::Random::RandomInt(int min, int max){
	int d = max - min + 1;
	return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// 函数作用：执行 UnrepeatedRandomizer 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - min：下限参数。
//   - max：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer(int min, int max)
{
  if(min <= max)
  {
    m_min = min;
    m_max = max;
  }
  else
  {
    m_min = max;
    m_max = min;
  }

  createValues();
}

// ---------------------------------------------------------------------------

// 函数作用：执行 UnrepeatedRandomizer 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - rnd：rnd 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer
  (const DUtils::Random::UnrepeatedRandomizer& rnd)
{
  *this = rnd;
}

// ---------------------------------------------------------------------------

// 函数作用：获取 get 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
int DUtils::Random::UnrepeatedRandomizer::get()
{
  if(empty()) createValues();
  
  DUtils::Random::SeedRandOnce();
  
  int k = DUtils::Random::RandomInt(0, m_values.size()-1);
  int ret = m_values[k];
  m_values[k] = m_values.back();
  m_values.pop_back();
  
  return ret;
}

// ---------------------------------------------------------------------------

// 函数作用：创建 createValues 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void DUtils::Random::UnrepeatedRandomizer::createValues()
{
  int n = m_max - m_min + 1;
  
  m_values.resize(n);
  for(int i = 0; i < n; ++i) m_values[i] = m_min + i;
}

// ---------------------------------------------------------------------------

// 函数作用：执行 reset 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void DUtils::Random::UnrepeatedRandomizer::reset()
{
  if((int)m_values.size() != m_max - m_min + 1) createValues();
}

// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer& 
DUtils::Random::UnrepeatedRandomizer::operator=
  (const DUtils::Random::UnrepeatedRandomizer& rnd)
{
  if(this != &rnd)
  {
    this->m_min = rnd.m_min;
    this->m_max = rnd.m_max;
    this->m_values = rnd.m_values;
  }
  return *this;
}

// ---------------------------------------------------------------------------


