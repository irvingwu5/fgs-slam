// 文件作用：验证原生 LSD 接口；隶属于points_lsd 原生测试模块。
#include <iostream>
#include <opencv2/opencv.hpp>

#include "lsd.h"

// 函数作用：解析参数并运行当前演示入口。
// 所属模块：points_lsd 原生测试。
// 输入：
//   - argc：命令行参数数量。
//   - argv：命令行参数数组。
// 输出：返回声明类型规定的检测、数值或状态结果；void 函数通过输出参数或资源状态产生结果。
int main(int argc, char **argv){
    std::cout << "**********************************************" << std::endl;
    std::cout << "****************** TLSD Test *****************" << std::endl;
    std::cout << "**********************************************" << std::endl;

    cv::Mat gray = cv::imread("../resources/ai_001_001.frame.0000.color.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img_flt;
    gray.convertTo(img_flt, CV_64FC1);

    double *imagePtr = reinterpret_cast<double *>(img_flt.data);

    // LSD call. Returns [x1,y1,x2,y2,width,p,-log10(NFA)] for each segment
    int N;
    double *out = lsd(&N, imagePtr, img_flt.cols, img_flt.rows);

    cv::Mat color;
    cv::cvtColor(gray,color, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < N; i++) {
        cv::line(color,
                 cv::Point(out[7 * i + 0], out[7 * i + 1]),
                 cv::Point(out[7 * i + 2], out[7 * i + 3]), CV_RGB(0, 255, 0));
    }
    free((void *) out);


    cv::imshow("segments", color);
    cv::waitKey();
}