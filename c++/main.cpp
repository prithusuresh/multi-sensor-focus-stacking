#include "clone.h"
#include <opencv2/opencv.hpp>

void naiveClone(cv::InputArray background_,
                cv::InputArray foreground_,
                cv::InputArray foregroundMask_,
                int offsetX, int offsetY,
                cv::OutputArray destination_)
{
    cv::Mat bg = background_.getMat();
    cv::Mat fg = foreground_.getMat();
    cv::Mat fgm = foregroundMask_.getMat();

    destination_.create(bg.size(), bg.type());
    cv::Mat dst = destination_.getMat();

    cv::Rect overlapAreaBg, overlapAreaFg;
    blend::detail::findOverlap(background_, foreground_, offsetX, offsetY, overlapAreaBg, overlapAreaFg);

    bg.copyTo(dst);
    fg(overlapAreaFg).copyTo(dst(overlapAreaBg), fgm(overlapAreaFg));

}


int main(int argc, char **argv)
{


    cv::Mat background = cv::imread("dataset/0(new).png");
    cv::Mat foreground = cv::imread("dataset/rsz_1new.png");
    cv::Mat mask = cv::imread("dataset/rsz_mask_black.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    int offsetx = 0;
    int offsety = 0;


    cv::Mat result;

    naiveClone(background, foreground, mask, offsetx, offsety, result);
    cv::imshow("Naive", result);
    cv::imwrite("naive.png", result);

    cv::waitKey();

    return 0;
}




