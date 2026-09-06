// 文件作用：提取 BRIEF 特征并验证候选几何一致性；隶属于 BRIEF 特征与几何验证 模块。
/**
 * File: BRIEF.cpp
 * Author: Dorian Galvez
 * Date: September 2010
 * Description: implementation of BRIEF (Binary Robust Independent 
 *   Elementary Features) descriptor by 
 *   Michael Calonder, Vincent Lepetitand Pascal Fua
 *   + close binary tests (by Dorian Galvez-Lopez)
 * License: see the LICENSE.txt file
 *
 */

#include "BRIEF.h"
#include "DUtils.h"
#include <boost/dynamic_bitset.hpp>
#include <vector>

using namespace std;
using namespace DVision;

// ----------------------------------------------------------------------------

// 函数作用：执行 BRIEF 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - nbits：nbits 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - patch_size：patch_size 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - type：type 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BRIEF::BRIEF(int nbits, int patch_size, Type type):
  m_bit_length(nbits), m_patch_size(patch_size), m_type(type)
{
  assert(patch_size > 1);
  assert(nbits > 0);
  generateTestPoints();
}

// ----------------------------------------------------------------------------

// 函数作用：释放对象持有的资源。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
BRIEF::~BRIEF()
{
}

// ---------------------------------------------------------------------------

// 函数作用：在关键点位置计算 BRIEF 二进制描述子。
// 所属模块：BRIEF 特征与几何验证。
// 输入：
//   - image：输入图像。
//   - points：points 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - descriptors：BRIEF 描述子集合。
//   - treat_image：输入图像或图像集合。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BRIEF::compute(const cv::Mat &image, 
    const std::vector<cv::KeyPoint> &points,
    vector<bitset> &descriptors,
    bool treat_image) const
{
  const float sigma = 2.f;
  const cv::Size ksize(9, 9);
  
  cv::Mat im;
  if(treat_image)
  {
    cv::Mat aux;
    if(image.depth() == 3)
    {
      cv::cvtColor(image, aux, CV_RGB2GRAY);
    }
    else
    {
      aux = image;
    }

    cv::GaussianBlur(aux, im, ksize, sigma, sigma);
    
  }
  else
  {
    im = image;
  }
  
  assert(im.type() == CV_8UC1);
  assert(im.isContinuous());
  
  // use im now
  const int W = im.cols;
  const int H = im.rows;
  
  descriptors.resize(points.size());
  std::vector<bitset>::iterator dit;

  std::vector<cv::KeyPoint>::const_iterator kit;
  
  int x1, y1, x2, y2;
  
  dit = descriptors.begin();
  for(kit = points.begin(); kit != points.end(); ++kit, ++dit)
  {
    dit->resize(m_bit_length);
    dit->reset();

    for(unsigned int i = 0; i < m_x1.size(); ++i)
    {
      x1 = (int)(kit->pt.x + m_x1[i]);
      y1 = (int)(kit->pt.y + m_y1[i]);
      x2 = (int)(kit->pt.x + m_x2[i]);
      y2 = (int)(kit->pt.y + m_y2[i]);
      
      if(x1 >= 0 && x1 < W && y1 >= 0 && y1 < H 
        && x2 >= 0 && x2 < W && y2 >= 0 && y2 < H)
      {
        if( im.ptr<unsigned char>(y1)[x1] < im.ptr<unsigned char>(y2)[x2] )
        {
          dit->set(i);
        }        
      } // if (x,y)_1 and (x,y)_2 are in the image
            
    } // for each (x,y)
  } // for each keypoint
}

// ---------------------------------------------------------------------------

// 函数作用：执行 generateTestPoints 对应的构造、计算或状态操作。
// 所属模块：BRIEF 特征与几何验证。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
void BRIEF::generateTestPoints()
{  
  m_x1.resize(m_bit_length);
  m_y1.resize(m_bit_length);
  m_x2.resize(m_bit_length);
  m_y2.resize(m_bit_length);

  const float g_mean = 0.f;
  const float g_sigma = 0.2f * (float)m_patch_size;
  const float c_sigma = 0.08f * (float)m_patch_size;
  
  float sigma2;
  if(m_type == RANDOM)
    sigma2 = g_sigma;
  else
    sigma2 = c_sigma;
  
  const int max_v = m_patch_size / 2;
  
  DUtils::Random::SeedRandOnce();
  
  for(int i = 0; i < m_bit_length; ++i)
  {
    int x1, y1, x2, y2;
    
    do
    {
      x1 = DUtils::Random::RandomGaussianValue(g_mean, g_sigma);
    } while( x1 > max_v || x1 < -max_v);
    
    do
    {
      y1 = DUtils::Random::RandomGaussianValue(g_mean, g_sigma);
    } while( y1 > max_v || y1 < -max_v);
    
    float meanx, meany;
    if(m_type == RANDOM)
      meanx = meany = g_mean;
    else
    {
      meanx = x1;
      meany = y1;
    }
    
    do
    {
      x2 = DUtils::Random::RandomGaussianValue(meanx, sigma2);
    } while( x2 > max_v || x2 < -max_v);
    
    do
    {
      y2 = DUtils::Random::RandomGaussianValue(meany, sigma2);
    } while( y2 > max_v || y2 < -max_v);
    
    m_x1[i] = x1;
    m_y1[i] = y1;
    m_x2[i] = x2;
    m_y2[i] = y2;
  }

}

// ----------------------------------------------------------------------------

