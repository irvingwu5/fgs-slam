// 文件作用：提供随机数、时间戳和异常等通用能力；隶属于 基础工具 模块。
/*
 * File: Timestamp.cpp
 * Author: Dorian Galvez-Lopez
 * Date: March 2009
 * Description: timestamping functions
 * 
 * Note: in windows, this class has a 1ms resolution
 *
 * License: see the LICENSE.txt file
 *
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#ifndef WIN32
#define WIN32
#endif
#endif

#ifdef WIN32
#include <sys/timeb.h>
#define sprintf sprintf_s
#else
#include <sys/time.h>
#endif

#include "Timestamp.h"

using namespace std;

using namespace DUtils;

// 函数作用：执行 Timestamp 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - option：option 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
Timestamp::Timestamp(Timestamp::tOptions option)
{
  if(option & CURRENT_TIME)
    setToCurrentTime();
  else if(option & ZERO)
    setTime(0.);
}

// 函数作用：释放对象持有的资源。
// 所属模块：基础工具。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
Timestamp::~Timestamp(void)
{
}

// 函数作用：判断是否为空 empty 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
bool Timestamp::empty() const
{
  return m_secs == 0 && m_usecs == 0;
}

// 函数作用：设置 setToCurrentTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void Timestamp::setToCurrentTime(){
	
#ifdef WIN32
	struct __timeb32 timebuffer;
	_ftime32_s( &timebuffer ); // C4996
	// Note: _ftime is deprecated; consider using _ftime_s instead
	m_secs = timebuffer.time;
	m_usecs = timebuffer.millitm * 1000;
#else
	struct timeval now;
	gettimeofday(&now, NULL);
	m_secs = now.tv_sec;
	m_usecs = now.tv_usec;
#endif

}

// 函数作用：设置 setTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：
//   - stime：stime 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void Timestamp::setTime(const string &stime){
	string::size_type p = stime.find('.');
	if(p == string::npos){
		m_secs = atol(stime.c_str());
		m_usecs = 0;
	}else{
		m_secs = atol(stime.substr(0, p).c_str());
		
		string s_usecs = stime.substr(p+1, 6);
		m_usecs = atol(stime.substr(p+1).c_str());
		m_usecs *= (unsigned long)pow(10.0, double(6 - s_usecs.length()));
	}
}

// 函数作用：设置 setTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：
//   - s：s 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void Timestamp::setTime(double s)
{
  m_secs = (unsigned long)s;
  m_usecs = (s - (double)m_secs) * 1e6;
}

// 函数作用：获取 getFloatTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
double Timestamp::getFloatTime() const {
	return double(m_secs) + double(m_usecs)/1000000.0;
}

// 函数作用：获取 getStringTime 对应的数据或状态。
// 所属模块：基础工具。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
string Timestamp::getStringTime() const {
	char buf[32];
	sprintf(buf, "%.6lf", this->getFloatTime());
	return string(buf);
}

double Timestamp::operator- (const Timestamp &t) const {
	return this->getFloatTime() - t.getFloatTime();
}

Timestamp& Timestamp::operator+= (double s)
{
  *this = *this + s;
  return *this;
}

Timestamp& Timestamp::operator-= (double s)
{
  *this = *this - s;
  return *this;
}

Timestamp Timestamp::operator+ (double s) const
{
	unsigned long secs = (long)floor(s);
	unsigned long usecs = (long)((s - (double)secs) * 1e6);

  return this->plus(secs, usecs);
}

// 函数作用：执行 plus 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - secs：secs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - usecs：usecs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
Timestamp Timestamp::plus(unsigned long secs, unsigned long usecs) const
{
  Timestamp t;

	const unsigned long max = 1000000ul;

	if(m_usecs + usecs >= max)
		t.setTime(m_secs + secs + 1, m_usecs + usecs - max);
	else
		t.setTime(m_secs + secs, m_usecs + usecs);
	
	return t;
}

Timestamp Timestamp::operator- (double s) const
{
	unsigned long secs = (long)floor(s);
	unsigned long usecs = (long)((s - (double)secs) * 1e6);

	return this->minus(secs, usecs);
}

// 函数作用：执行 minus 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - secs：secs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - usecs：usecs 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
Timestamp Timestamp::minus(unsigned long secs, unsigned long usecs) const
{
  Timestamp t;

	const unsigned long max = 1000000ul;

	if(m_usecs < usecs)
		t.setTime(m_secs - secs - 1, max - (usecs - m_usecs));
	else
		t.setTime(m_secs - secs, m_usecs - usecs);
	
	return t;
}

bool Timestamp::operator> (const Timestamp &t) const
{
	if(m_secs > t.m_secs) return true;
	else if(m_secs == t.m_secs) return m_usecs > t.m_usecs;
	else return false;
}

bool Timestamp::operator>= (const Timestamp &t) const
{
	if(m_secs > t.m_secs) return true;
	else if(m_secs == t.m_secs) return m_usecs >= t.m_usecs;
	else return false;
}

bool Timestamp::operator< (const Timestamp &t) const
{
	if(m_secs < t.m_secs) return true;
	else if(m_secs == t.m_secs) return m_usecs < t.m_usecs;
	else return false;
}

bool Timestamp::operator<= (const Timestamp &t) const
{
	if(m_secs < t.m_secs) return true;
	else if(m_secs == t.m_secs) return m_usecs <= t.m_usecs;
	else return false;
}

bool Timestamp::operator== (const Timestamp &t) const
{
	return(m_secs == t.m_secs && m_usecs == t.m_usecs);
}


// 函数作用：执行 Format 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - machine_friendly：machine_friendly 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
string Timestamp::Format(bool machine_friendly) const 
{
  struct tm tm_time;

  time_t t = (time_t)getFloatTime();

#ifdef WIN32
  localtime_s(&tm_time, &t);
#else
  localtime_r(&t, &tm_time);
#endif
  
  char buffer[128];
  
  if(machine_friendly)
  {
    strftime(buffer, 128, "%Y%m%d_%H%M%S", &tm_time);
  }
  else
  {
    strftime(buffer, 128, "%c", &tm_time); // Thu Aug 23 14:55:02 2001
  }
  
  return string(buffer);
}

// 函数作用：执行 Format 对应的构造、计算或状态操作。
// 所属模块：基础工具。
// 输入：
//   - s：s 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
string Timestamp::Format(double s) {
	int days = int(s / (24. * 3600.0));
	s -= days * (24. * 3600.0);
	int hours = int(s / 3600.0);
	s -= hours * 3600;
	int minutes = int(s / 60.0);
	s -= minutes * 60;
	int seconds = int(s);
	int ms = int((s - seconds)*1e6);

	stringstream ss;
	ss.fill('0');
	bool b;
	if((b = (days > 0))) ss << days << "d ";
	if((b = (b || hours > 0))) ss << setw(2) << hours << ":";
	if((b = (b || minutes > 0))) ss << setw(2) << minutes << ":";
	if(b) ss << setw(2);
	ss << seconds;
	if(!b) ss << "." << setw(6) << ms;

	return ss.str();
}


