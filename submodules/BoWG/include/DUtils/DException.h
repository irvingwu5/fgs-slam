// 文件作用：提供随机数、时间戳和异常等通用能力；隶属于 基础工具 模块。
/*	
 * File: DException.h
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: October 6, 2009
 * Description: general exception of the library
 * License: see the LICENSE.txt file
 *
 */

#pragma once

#ifndef __D_EXCEPTION__
#define __D_EXCEPTION__

#include <stdexcept>
#include <string>
using namespace std;

namespace DUtils {

/// General exception
class DException :
	public exception
{
public:
	/**
	 * Creates an exception with a general error message
	 */
// 函数作用：执行 DException 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
	DException(void) throw(): m_message("DUtils exception"){}

	/**
	 * Creates an exception with a custom error message
	 * @param msg: message
	 */
// 函数作用：执行 DException 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - msg：msg 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
	DException(const char *msg) throw(): m_message(msg){}
	
	/**
	 * Creates an exception with a custom error message
	 * @param msg: message
	 */
// 函数作用：执行 DException 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - msg：msg 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
	DException(const string &msg) throw(): m_message(msg){}

  /**
	 * Destructor
	 */
// 函数作用：释放对象持有的资源。
// 所属模块：基础工具。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
	virtual ~DException(void) throw(){}

	/**
	 * Returns the exception message
	 */
// 函数作用：执行 what 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
	virtual const char* what() const throw()
	{
		return m_message.c_str();
	}

protected:
  /// Error message
	string m_message;
};

}

#endif

