#include <cstring>

#include "CLI11.hpp"
#include "deepstream_sample_parameters.h"

static parameters params;

parameters *parse_parameters(int argc, char **argv)
{
    memset(&params, 0, sizeof(parameters));

    CLI::App app("DeepStream SDK Sample Application");
    app.add_option("-w,--width", params.width, "Width of the input video")->required();
    app.add_option("-e,--height", params.height, "Height of the input video")->required();
    app.add_option("-W,--display-width", params.display_width, "Width of the tile")->default_val(1920);
    app.add_option("-H,--display-height", params.display_height, "Height of the tile")->default_val(1080);
    app.add_option("-m,--model", params.model, "Path to the model file");
    app.add_option("-i,--input", params.input_urls, "Path to the input video file")->required();
    app.add_option("-l,--lib", params.lib, "Path to the custom library");
    app.add_option("-f,--force-process", params.force_process, "Force process even if no pre/post buffer change")->default_val(true);
    app.add_option("--osd-mode", params.osd_mode, "OSD mode (0: CPU, 1: GPU, 2: HW)")->default_val(NvOSD_Mode::MODE_GPU);
    app.add_flag("--debug,--no-debug", params.debug_mode, "Enable debug mode");
    if (argc < 2) {
        std::cout << app.help() << std::endl;
        exit(-1);
    } else {
        try {
            (app).parse((argc), (argv));
        } catch(const CLI::ParseError &e) {
            (app).exit(e);
            exit(-1);
        }
        params.initialized = true;
    }
    if (params.lib.empty()) {
        params.lib = "./libcustom_videoimpl.so";
    }

    return &params;
}
