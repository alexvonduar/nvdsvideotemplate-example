/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <cassert>

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
//#include "nvds_yml_parser.h"

#include "deepstream_sample_parameters.h"


#define MAKE_GST_ELEMENT(element, factory, name) \
    do { \
        element = gst_element_factory_make(factory, name); \
        if (!element) { \
            g_printerr("Unable to create gst element %s as " #element "(%s)", factory, name); \
            return -1; \
        } \
    } while (0)

#define LINK_GST_ELEMENT(src, sink) \
    do { \
        if (!gst_element_link(elements[src].second, elements[sink].second)) { \
            g_printerr("Unable to link gst element %s(%s) to %s(%s)", src, elements[src].first.c_str(), sink, elements[sink].first.c_str()); \
            return -1; \
        } \
    } while (0)

#define MAX_DISPLAY_LEN 64

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
typedef enum _PGIE_CLASS_ID {
    PGIE_CLASS_ID_VEHICLE = 0,
    PGIE_CLASS_ID_BICYCLE,
    PGIE_CLASS_ID_PERSON,
    PGIE_CLASS_ID_ROADSIGN,
    PGIE_CLASS_ID_MAX
} PGIE_CLASS_ID;

gchar pgie_classes_str[PGIE_CLASS_ID_MAX][32] = {
    "Vehicle", "Bicycle", "Person", "Roadsign"
};

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint obj_counter[PGIE_CLASS_ID_MAX];
    //guint vehicle_count = 0;
    //guint bicycle_count = 0;
    //guint person_count = 0;
    //guint road_sign = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    memset(obj_counter, 0, PGIE_CLASS_ID_MAX * sizeof(guint));

    std::string display_text;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            obj_counter[obj_meta->class_id]++;
            obj_meta->rect_params.border_color.red=0.0;
            obj_meta->rect_params.border_color.green=0.0;
            obj_meta->rect_params.border_color.blue=1.0;
            obj_meta->rect_params.border_color.alpha=0.8;
            num_rects++;
        }
        display_text = "Frame Number: " + std::to_string(frame_number) +
            " Total: " + std::to_string(num_rects) +
            " Vehicle: " + std::to_string(obj_counter[PGIE_CLASS_ID_VEHICLE]) +
            " Person: " + std::to_string(obj_counter[PGIE_CLASS_ID_PERSON]);
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = (char *)(g_malloc0 (display_text.length() + 1));
        //memset(txt_params->display_text, 0, display_text.length() + 1);
        memcpy(txt_params->display_text, display_text.c_str(), display_text.length());
        //offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        //offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    //g_print ("Frame Number = %d Number of objects = %d "
    //        "Vehicle Count = %d Person Count = %d\n",
    //        frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
    GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
    if (!caps) {
        caps = gst_pad_query_caps (decoder_src_pad, NULL);
    }
    const GstStructure *str = gst_caps_get_structure (caps, 0);
    const gchar *name = gst_structure_get_name (str);
    GstElement *source_bin = (GstElement *) data;
    GstCapsFeatures *features = gst_caps_get_features (caps, 0);

    /* Need to check if the pad created by the decodebin is for video and not
    * audio. */
    if (!strncmp (name, "video", 5)) {
        /* Link the decodebin pad only if decodebin has picked nvidia
        * decoder plugin nvdec_*. We do this by checking if the pad caps contain
        * NVMM memory features. */
        if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
            /* Get the source bin ghost pad */
            GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
            if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
                    decoder_src_pad)) {
                g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
            }
            gst_object_unref (bin_ghost_pad);
        } else {
            g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
        }
    }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
    g_print ("Decodebin child added: %s\n", name);
    if (g_strrstr (name, "decodebin") == name) {
        g_signal_connect (G_OBJECT (object), "child-added",
            G_CALLBACK (decodebin_child_added), user_data);
    }
    if (g_strrstr (name, "source") == name) {
            g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
    }
}

static int
create_source_bin (guint index, const gchar * const uri, std::string& name, std::pair<std::string,GstElement*>& bin)
{
    std::string input_uri(uri);
    if (input_uri.find("/") == 0) {
        printf ("absolut file source\n");
        input_uri = "file://" + std::string(uri);
    } else if (input_uri.find("./") == 0) {
        printf ("relevant file source\n");
        std::filesystem::path p = uri;
        input_uri = "file://" + std::filesystem::absolute(p).string();
    }
    bin.second = nullptr;
    GstElement *uri_decode_bin = nullptr;
    gchar bin_name[16] = { };

    g_snprintf (bin_name, 15, "source-bin-%02d", index);
    /* Create a source GstBin to abstract this bin's content from the rest of the
    * pipeline */
    bin.second = gst_bin_new (bin_name);
    name = bin_name;

    /* Source element for reading from the uri.
    * We will use decodebin and let it figure out the container format of the
    * stream and the codec and plug the appropriate demux and decode plugins. */
    bin.first = "uridecodebin";

    //if (PERF_MODE) {
    //    uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
    //    g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
    //    g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);
    //} else {
    //    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
    //}
    uri_decode_bin = gst_element_factory_make(bin.first.c_str(), "uridecodebin");

    if (bin.second == nullptr or uri_decode_bin == nullptr) {
        g_printerr ("One element in source bin could not be created.\n");
        return -1;
    }

    /* We set the input uri to the source element */
    g_object_set (G_OBJECT (uri_decode_bin), "uri", input_uri.c_str(), NULL);

    /* Connect to the "pad-added" signal of the decodebin which generates a
    * callback once a new pad for raw data has beed created by the decodebin */
    g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
        G_CALLBACK (cb_newpad), bin.second);
    g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
        G_CALLBACK (decodebin_child_added), bin.second);

    gst_bin_add (GST_BIN (bin.second), uri_decode_bin);

    /* We need to create a ghost pad for the source bin which will act as a proxy
    * for the video decoder src pad. The ghost pad will not have a target right
    * now. Once the decode bin creates the video decoder and generates the
    * cb_newpad callback, we will set the ghost pad target to the video decoder
    * src pad. */
    if (!gst_element_add_pad (bin.second, gst_ghost_pad_new_no_target ("src",
                GST_PAD_SRC))) {
        g_printerr ("Failed to add ghost pad in source bin\n");
        return -1;
    }

    return 0;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n",
                   GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
            g_printerr("Error details: %s\n", debug);
        g_free(debug);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

int main(int argc, char *argv[])
{
    GMainLoop *loop = nullptr;
    GstElement *pipeline = nullptr;
    GstBus *bus = nullptr;

    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    /* Check input arguments */
    parameters * params = parse_parameters(argc, argv);
    if (not params->initialized) {
        return -1;
    }
    const auto num_sources = params->input_urls.size();

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new("dsexample-pipeline");
    if (pipeline == nullptr)
    {
        g_printerr("Unable to create pipeline\n");
        return -1;
    }

    std::unordered_map<std::string, std::pair<std::string, GstElement *>> elements =
    {
        {"streammux", {"nvstreammux", nullptr}},
        {"converter_prepgie", {"nvvideoconvert", nullptr}},
        {"caps_prepgie", {"capsfilter", nullptr}},
        {"pgie", {"nvdsvideotemplate", nullptr}},
        {"converter_postpgie", {"nvvideoconvert", nullptr}},
        {"caps_preosd", {"capsfilter", nullptr}},
        {"osd", {"nvdsosd", nullptr}},
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
        {"transform", {"nvegltransform", nullptr}},
#endif
        {"sink", {"nveglglessink", nullptr}}
    };

    if (params->input_urls.size() > 1) {
        elements.emplace("tiler", std::make_pair("nvmultistreamtiler", nullptr));
    }

    for (auto & e : elements)
    {
        MAKE_GST_ELEMENT(e.second.second, e.second.first.c_str(), e.first.c_str());
    }

    g_object_set(
        G_OBJECT(elements["streammux"].second),
        "width", params->width,
        "height", params->height,
        "batch-size", num_sources,
        "batched-push-timeout", 4000000,
        NULL
    );

    if (not params->lib.empty()) {
        g_object_set(G_OBJECT(elements["pgie"].second), "customlib-name", params->lib.c_str(), NULL);
    }

    if (not params->model.empty()) {
        g_object_set(G_OBJECT(elements["pgie"].second),
            "customlib-props", (std::string("model-path:")+params->model).c_str(), NULL);
    }

    g_object_set(G_OBJECT(elements["pgie"].second),
        "customlib-props", (std::string("force:")+std::to_string(params->force_process)).c_str(), NULL);

    g_object_set(G_OBJECT(elements["pgie"].second),
        "customlib-props", "key1:value1",
        "customlib-props", "key2:value2",
        NULL);

#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
    //g_object_set(G_OBJECT(nvvidconv), "nvbuf-memory-type", 4, "bl-output", 0, NULL);
#endif

    const std::string caps_prepgie_str = "video/x-raw(memory:NVMM), format=NV12, block-linear=false";
    g_object_set(G_OBJECT(elements["caps_prepgie"].second), "caps", gst_caps_from_string(caps_prepgie_str.c_str()), NULL);

    std::string caps_preosd_str = "video/x-raw(memory:NVMM), block-linear=false";//, format=RGBA;
    if (params->osd_mode != NvOSD_Mode::MODE_GPU) {
        caps_preosd_str += ", format=RGBA";
    } else {
        caps_preosd_str += ", format=NV12";
    }

    if (num_sources == 1) {
        caps_preosd_str +=  ", width=" + std::to_string(params->display_width) +
                            ", height=" + std::to_string(params->display_height);
    }
    g_object_set(G_OBJECT(elements["caps_preosd"].second), "caps", gst_caps_from_string(caps_preosd_str.c_str()), NULL);
    g_object_set(G_OBJECT(elements["osd"].second), "process-mode", params->osd_mode, NULL);

    if (num_sources > 1) {
        g_object_set(G_OBJECT(elements["tiler"].second),
            "width", params->display_width,
            "height", params->display_height,
            NULL);
    }

    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    if (num_sources == 1) {
        osd_sink_pad = gst_element_get_static_pad(elements["osd"].second, "sink");
    } else {
        osd_sink_pad = gst_element_get_static_pad(elements["tiler"].second, "sink");
    }
    assert(osd_sink_pad);
    if (!osd_sink_pad)
        g_print("Unable to get osd sink pad\n");
    else
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                            osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(osd_sink_pad);

    g_object_set(G_OBJECT(elements["sink"].second), "sync", 0, NULL);

    /* we add all elements into the pipeline */
    for (auto & e : elements)
    {
        if (e.second.second != nullptr)
        {
            gst_bin_add(GST_BIN(pipeline), e.second.second);
        }
    }

    std::unordered_map<std::string, std::pair<std::string, GstElement *>> source_bins;

    /* we set the input filename to the source element */
    for (int i = 0; i < num_sources; i++) {
        std::string source_bin_name;
        std::pair<std::string, GstElement *> source_bin;
        GstPad *sinkpad, *srcpad;
        gchar pad_name[16] = { };

        if (create_source_bin(i, (gchar *)(params->input_urls[i].c_str()), source_bin_name, source_bin) < 0) {
            g_printerr("Failed to create source bin. Exiting.\n");
            return -1;
        }

        if (source_bin.second == nullptr) {
            g_printerr("Failed to create source bin. Exiting.\n");
            return -1;
        }
        source_bins[source_bin_name] = source_bin;

        gst_bin_add (GST_BIN (pipeline), source_bin.second);

        g_snprintf (pad_name, 15, "sink_%u", i);
        sinkpad = gst_element_get_request_pad (elements["streammux"].second, pad_name);
        if (!sinkpad) {
            g_printerr ("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }

        srcpad = gst_element_get_static_pad (source_bin.second, "src");
        if (!srcpad) {
            g_printerr ("Failed to get src pad of source bin. Exiting.\n");
            return -1;
        }

        if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
            return -1;
        }

        gst_object_unref (srcpad);
        gst_object_unref (sinkpad);
    }

    /* we add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Set up the pipeline */

    /* link all the components */
    LINK_GST_ELEMENT("streammux", "converter_prepgie");
    LINK_GST_ELEMENT("converter_prepgie", "caps_prepgie");
    LINK_GST_ELEMENT("caps_prepgie", "pgie");
    LINK_GST_ELEMENT("pgie", "converter_postpgie");
    LINK_GST_ELEMENT("converter_postpgie", "caps_preosd");
    if (elements.find("tiler") != elements.end()) {
        LINK_GST_ELEMENT("caps_preosd", "tiler");
        LINK_GST_ELEMENT("tiler", "osd");
    } else {
        LINK_GST_ELEMENT("caps_preosd", "osd");
    }
    //LINK_GST_ELEMENT("caps_preosd", "osd");
    if (elements.find("transform") != elements.end()) {
        LINK_GST_ELEMENT("osd", "transform");
        LINK_GST_ELEMENT("transform", "sink");
    } else {
        LINK_GST_ELEMENT("osd", "sink");
    }

    /* Set the pipeline to "playing" state */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}
