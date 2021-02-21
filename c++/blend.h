

#ifndef POISSON_BLEND_H
#define POISSON_BLEND_H

#include <opencv2/core/core.hpp>

namespace blend {

    void seamlessBlend(cv::InputArray first,cv::InputArray second,cv::InputArray mask,cv::OutputArray destination);

}
#endif
