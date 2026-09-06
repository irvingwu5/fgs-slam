// 文件作用：提供随机数、时间戳和异常等通用能力；隶属于 基础工具 模块。
/*
 * File: Timestamp.h
 * Author: Dorian Galvez-Lopez
 * Date: March 2009
 * Description: timestamping functions
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_TIMESTAMP__
#define __D_TIMESTAMP__

#include <iostream>
using namespace std;

namespace DUtils {

/// Timestamp
class Timestamp
{
public:

  /// Options to initiate a timestamp
  enum tOptions
  {
    NONE = 0,
    CURRENT_TIME = 0x1,
    ZERO = 0x2
  };
  
public:
  
  /**
   * Creates a timestamp 
   * @param option option to set the initial time stamp
   */
// 函数作用：执行 Timestamp 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - option：option 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
	Timestamp(Timestamp::tOptions option = NONE);
	
	/**
	 * Destructor
	 */
// 函数作用：释放对象持有的资源。
// 所属模块：基础工具。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
	virtual ~Timestamp(void);

  /**
   * Says if the timestamp is "empty": seconds and usecs are both 0, as 
   * when initiated with the ZERO flag
   * @return true iif secs == usecs == 0
   */
// 函数作用：判断是否为空 empty 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  bool empty() const;

	/**
	 * Sets this instance to the current time
	 */
// 函数作用：设置 setToCurrentTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	void setToCurrentTime();

	/**
	 * Sets the timestamp from seconds and microseconds
	 * @param secs: seconds
	 * @param usecs: microseconds
	 */
// 函数作用：设置 setTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：
//   - secs：secs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - usecs：usecs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	inline void setTime(unsigned long secs, unsigned long usecs){
		m_secs = secs;
		m_usecs = usecs;
	}
	
	/**
	 * Returns the timestamp in seconds and microseconds
	 * @param secs seconds
	 * @param usecs microseconds
	 */
// 函数作用：获取 getTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：
//   - secs：secs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - usecs：usecs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	inline void getTime(unsigned long &secs, unsigned long &usecs) const
	{
	  secs = m_secs;
	  usecs = m_usecs;
	}

	/**
	 * Sets the timestamp from a string with the time in seconds
	 * @param stime: string such as "1235603336.036609"
	 */
// 函数作用：设置 setTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：
//   - stime：stime 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	void setTime(const string &stime);
	
	/**
	 * Sets the timestamp from a number of seconds from the epoch
	 * @param s seconds from the epoch
	 */
// 函数作用：设置 setTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：
//   - s：s 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	void setTime(double s);
	
	/**
	 * Returns this timestamp as the number of seconds in (long) float format
	 */
// 函数作用：获取 getFloatTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	double getFloatTime() const;

	/**
	 * Returns this timestamp as the number of seconds in fixed length string format
	 */
// 函数作用：获取 getStringTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	string getStringTime() const;

	/**
	 * Returns the difference in seconds between this timestamp (greater) and t (smaller)
	 * If the order is swapped, a negative number is returned
	 * @param t: timestamp to subtract from this timestamp
	 * @return difference in seconds
	 */
	double operator- (const Timestamp &t) const;

	/** 
	 * Returns a copy of this timestamp + s seconds + us microseconds
	 * @param s seconds
	 * @param us microseconds
	 */
// 函数作用：执行 plus 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - s：s 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - us：us 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	Timestamp plus(unsigned long s, unsigned long us) const;

  /**
   * Returns a copy of this timestamp - s seconds - us microseconds
   * @param s seconds
   * @param us microseconds
   */
// 函数作用：执行 minus 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - s：s 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - us：us 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  Timestamp minus(unsigned long s, unsigned long us) const;

  /**
   * Adds s seconds to this timestamp and returns a reference to itself
   * @param s seconds
   * @return reference to this timestamp
   */
  Timestamp& operator+= (double s);
  
  /**
   * Substracts s seconds to this timestamp and returns a reference to itself
   * @param s seconds
   * @return reference to this timestamp
   */
  Timestamp& operator-= (double s);

	/**
	 * Returns a copy of this timestamp + s seconds
	 * @param s: seconds
	 */
	Timestamp operator+ (double s) const;

	/**
	 * Returns a copy of this timestamp - s seconds
	 * @param s: seconds
	 */
	Timestamp operator- (double s) const;

	/**
	 * Returns whether this timestamp is at the future of t
	 * @param t
	 */
	bool operator> (const Timestamp &t) const;

	/**
	 * Returns whether this timestamp is at the future of (or is the same as) t
	 * @param t
	 */
	bool operator>= (const Timestamp &t) const;

	/** 
	 * Returns whether this timestamp and t represent the same instant
	 * @param t
	 */
	bool operator== (const Timestamp &t) const;

	/**
	 * Returns whether this timestamp is at the past of t
	 * @param t
	 */
	bool operator< (const Timestamp &t) const;

	/**
	 * Returns whether this timestamp is at the past of (or is the same as) t
	 * @param t
	 */
	bool operator<= (const Timestamp &t) const;

  /**
   * Returns the timestamp in a human-readable string
   * @param machine_friendly if true, the returned string is formatted
   *   to yyyymmdd_hhmmss, without weekday or spaces
   * @note This has not been tested under Windows
   * @note The timestamp is truncated to seconds
   */
// 函数作用：执行 Format 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - machine_friendly：machine_friendly 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
  string Format(bool machine_friendly = false) const;

	/**
	 * Returns a string version of the elapsed time in seconds, with the format
	 * xd hh:mm:ss, hh:mm:ss, mm:ss or s.us
	 * @param s: elapsed seconds (given by getFloatTime) to format
	 */
// 函数作用：执行 Format 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - s：s 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	static string Format(double s);
	

protected:
  /// Seconds
	unsigned long m_secs;	// seconds
	/// Microseconds
	unsigned long m_usecs;	// microseconds
};

}

#endif

