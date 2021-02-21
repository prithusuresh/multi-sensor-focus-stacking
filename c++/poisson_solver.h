

#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <opencv2/core/core.hpp>

namespace blend {

    namespace constants {
        const unsigned char UNKNOWN = 0;
        const unsigned char DIRICHLET_BD = 1;
        const unsigned char NEUMANN_BD = 2;
    }

    /**
        Solve multi-channel Poisson equations on rectangular domain.
    */
    void solvePoissonEquations(
        cv::InputArray f,
        cv::InputArray bdMask,
        cv::InputArray bdValues,
        cv::OutputArray result);

}
#endif
