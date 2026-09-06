// 文件作用：提供随机数、时间戳和异常等通用能力；隶属于 基础工具 模块。
/*	
 * File: Random.h
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010, November 2011
 * Description: manages pseudo-random numbers
 * License: see the LICENSE.txt file
 *
 */

#pragma once
#ifndef __D_RANDOM__
#define __D_RANDOM__

#include <cstdlib>
#include <vector>

namespace DUtils {

/// Functions to generate pseudo-random numbers
class Random
{
public:
  class UnrepeatedRandomizer;
  
public:
	/**
	 * Sets the random number seed to the current time
	 */
// 函数作用：执行 SeedRand 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static void SeedRand();
	
	/**
	 * Sets the random number seed to the current time only the first
	 * time this function is called
	 */
// 函数作用：执行 SeedRandOnce 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static void SeedRandOnce();

	/** 
	 * Sets the given random number seed
	 * @param seed
	 */
// 函数作用：执行 SeedRand 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - seed：随机数种子。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static void SeedRand(int seed);

	/** 
	 * Sets the given random number seed only the first time this function 
	 * is called
	 * @param seed
	 */
// 函数作用：执行 SeedRandOnce 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - seed：随机数种子。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static void SeedRandOnce(int seed);

	/**
	 * Returns a random number in the range [0..1]
	 * @return random T number in [0..1]
	 */
	template <class T>
// 函数作用：执行 RandomValue 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static T RandomValue(){
		return (T)rand()/(T)RAND_MAX;
	}

	/**
	 * Returns a random number in the range [min..max]
	 * @param min
	 * @param max
	 * @return random T number in [min..max]
	 */
	template <class T>
// 函数作用：执行 RandomValue 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - min：下限参数。
//   - max：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static T RandomValue(T min, T max){
		return Random::RandomValue<T>() * (max - min) + min;
	}

	/**
	 * Returns a random int in the range [min..max]
	 * @param min
	 * @param max
	 * @return random int in [min..max]
	 */
// 函数作用：执行 RandomInt 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - min：下限参数。
//   - max：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static int RandomInt(int min, int max);
	
	/** 
	 * Returns a random number from a gaussian distribution
	 * @param mean
	 * @param sigma standard deviation
	 */
	template <class T>
// 函数作用：执行 RandomGaussianValue 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - mean：mean 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - sigma：sigma 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static T RandomGaussianValue(T mean, T sigma)
	{
    // Box-Muller transformation
    T x1, x2, w, y1;

    do {
      x1 = (T)2. * RandomValue<T>() - (T)1.;
      x2 = (T)2. * RandomValue<T>() - (T)1.;
      w = x1 * x1 + x2 * x2;
    } while ( w >= (T)1. || w == (T)0. );

    w = sqrt( ((T)-2.0 * log( w ) ) / w );
    y1 = x1 * w;

    return( mean + y1 * sigma );
	}

private:

  /// If SeedRandOnce() or SeedRandOnce(int) have already been called
  static bool m_already_seeded;
  
};

// ---------------------------------------------------------------------------

/// Provides pseudo-random numbers with no repetitions
class Random::UnrepeatedRandomizer
{
public:

  /** 
   * Creates a randomizer that returns numbers in the range [min, max]
   * @param min
   * @param max
   */
// 函数作用：执行 UnrepeatedRandomizer 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - min：下限参数。
//   - max：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  UnrepeatedRandomizer(int min, int max);
// 函数作用：释放对象持有的资源。
// 所属模块：基础工具。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
  ~UnrepeatedRandomizer(){}
  
  /**
   * Copies a randomizer
   * @param rnd
   */
// 函数作用：执行 UnrepeatedRandomizer 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - rnd：rnd 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  UnrepeatedRandomizer(const UnrepeatedRandomizer& rnd);
  
  /**
   * Copies a randomizer
   * @param rnd
   */
  UnrepeatedRandomizer& operator=(const UnrepeatedRandomizer& rnd);
  
  /** 
   * Returns a random number not given before. If all the possible values
   * were already given, the process starts again
   * @return unrepeated random number
   */
// 函数作用：获取 get 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  int get();
  
  /**
   * Returns whether all the possible values between min and max were
   * already given. If get() is called when empty() is true, the behaviour
   * is the same than after creating the randomizer
   * @return true iff all the values were returned
   */
// 函数作用：判断是否为空 empty 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  inline bool empty() const { return m_values.empty(); }
  
  /**
   * Returns the number of values still to be returned
   * @return amount of values to return
   */
// 函数作用：执行 left 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  inline unsigned int left() const { return m_values.size(); }
  
  /**
   * Resets the randomizer as it were just created
   */
// 函数作用：执行 reset 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  void reset();
  
protected:

  /** 
   * Creates the vector with available values
   */
// 函数作用：创建 createValues 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  void createValues();

protected:

  /// Min of range of values
  int m_min;
  /// Max of range of values
  int m_max;

  /// Available values
  std::vector<int> m_values;

};

}

#endif

