#ifndef __DEEPSTREAM_SAMPLE_PARAMETERS_H__
#define __DEEPSTREAM_SAMPLE_PARAMETERS_H__

#include <string>
#include <vector>

#include "nvll_osd_struct.h"

typedef struct _parameters {
    bool initialized;
    int width;
    int height;
    int display_width;
    int display_height;
    NvOSD_Mode osd_mode;
    bool force_process;
    std::string model;
    std::string lib;
    std::vector<std::string> input_urls;
    bool debug_mode;
} parameters;

parameters *parse_parameters(int argc, char *argv[]);

#endif//__DEEPSTREAM_SAMPLE_PARAMETERS_H__
