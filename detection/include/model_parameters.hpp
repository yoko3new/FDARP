// Not present in upstream, but include/io.hpp includes it, so upstream does not
// compile without it. Only used by io::get_params(), which main.cpp does not call.
#ifndef MODEL_PARAMETERS_HPP
#define MODEL_PARAMETERS_HPP

struct ModelParameters
{
    double pos_gauss = 100.0;
    double neg_gauss = -100.0;
    int dilation_size = 10;
    double strength_threshold = 0.95;
    int size_threshold = 100;
    int gap_size = 4;
};

#endif
