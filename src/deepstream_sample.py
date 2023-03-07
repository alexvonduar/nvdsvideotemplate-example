#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
from common.bus_call import bus_call
from common.is_aarch_64 import is_aarch64
#import pyds
import platform
import math
import time
from ctypes import *
import gi
gi.require_version("Gst", "1.0")
#gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GLib
import configparser
import pyds

import argparse

MAX_DISPLAY_LEN = 64
#PGIE_CLASS_ID_VEHICLE = 0
#PGIE_CLASS_ID_BICYCLE = 1
#PGIE_CLASS_ID_PERSON = 2
#PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0
#pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

def parse_ds_version():
    global DS_VERSION_MAJOR
    DS_VERSION_MAJOR = 6
    global DS_VERSION_MINOR
    DS_VERSION_MINOR = 0
    global DS_VERSION_PATCH
    DS_VERSION_PATCH = 0
    with open("/opt/nvidia/deepstream/deepstream/version") as f:
        for line in f:
            if "version:" in line.lower():
                v = line.lower().replace("version:", "").strip().split(".")
                print("DeepStream version: ", v)
                if len(v) == 3:
                    DS_VERSION_MAJOR = int(v[0])
                    DS_VERSION_MINOR = int(v[1])
                    DS_VERSION_PATCH = int(v[2])
                    break
                elif (len(v) == 2):
                    DS_VERSION_MAJOR = int(v[0])
                    DS_VERSION_MINOR = int(v[1])
                    break
    global DS_VERSION_NUMBER
    DS_VERSION_NUMBER = (DS_VERSION_MAJOR * 1000) + (DS_VERSION_MINOR * 100) + DS_VERSION_PATCH

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        #Intiallizing object counter with 0.
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE:0,
            PGIE_CLASS_ID_PERSON:0,
            PGIE_CLASS_ID_BICYCLE:0,
            PGIE_CLASS_ID_ROADSIGN:0
        }
        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.8) #0.8 is alpha (opacity)
            try:
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if name.find("nvv4l2decoder") != -1:
        print("Setting ", name, " special properties\n")
        # 0 device 1 pinned 2 unified
        if is_aarch64():
            #Object.set_property("cudadec-memtype", 1)
            pass
        else:
            Object.set_property("cudadec-memtype", 0)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    if uri.startswith("/") or uri.startswith("./"):
        uri = "file://" + uri
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target(
            "src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Check input arguments
    number_sources = len(args)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Create nvstreammux instance to form batches from one or more sources.
    print("Creating streamux \n ")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    # Create nvvideoconvert after decoder
    print("Creating nvvideoconvert \n ")
    nvvidconv_postmux = Gst.ElementFactory.make("nvvideoconvert", "convertor_postmux")
    if not nvvidconv_postmux:
        sys.stderr.write(" Unable to create nvvideoconvert \n")
    if is_aarch64():
        nvvidconv_postmux.set_property("bl-output", 0)
        # 0 default 1 pinned 2 device 3 unified 4 surface array
        nvvidconv_postmux.set_property("nvbuf-memory-type", 4)
    else:
        nvvidconv_postmux.set_property("bl-output", 0)
        # 0 default 1 pinned 2 device 3 unified
        nvvidconv_postmux.set_property("nvbuf-memory-type", 3)

    # Create post mux caps filter
    cap_postmux = Gst.ElementFactory.make("capsfilter", "filter_postmux")
    if is_aarch64():
        cap_postmux.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12, block-linear=false"))
    else:
        cap_postmux.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12, block-linear=false"))

    print("Creating nvdsvideotemplate \n ")
    pgie = Gst.ElementFactory.make("nvdsvideotemplate", "inference")
    if not pgie:
        sys.stderr.write(" Unable to create nvdsvideotemplate \n")

    pgie.set_property("customlib-name", "./libcustomlib_videoimpl.so")
    pgie.set_property("customlib-props", "key1:value1")
    pgie.set_property("customlib-props", "key2:value2")
    pgie.set_property("customlib-props", "model-path:"+model_path)
    #pgie.set_property("customlib-props", "scale-factor:2")
    if DS_VERSION_NUMBER >= 6100:
        pgie.set_property("dummy-meta-insert", 1)
        pgie.set_property("fill-dummy-batch-meta", 0)

    # Create a caps filter
    caps_postpgie = Gst.ElementFactory.make("capsfilter", "filter_postpgie")
    if is_aarch64():
        caps_postpgie.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, block-linear=false"))
    else:
        caps_postpgie.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, block-linear=false"))

    print("Creating post pgie nvvidconv \n ")
    nvvidconv_postpgie = Gst.ElementFactory.make("nvvideoconvert", "convertor_postpgie")
    if not nvvidconv_postpgie:
        sys.stderr.write(" Unable to create post pgie nvvidconv \n")

    '''
    # Create a caps filter
    caps_postpgie = Gst.ElementFactory.make("capsfilter", "filter_postpgie")
    if is_aarch64():
        caps_postpgie.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, block-linear=false"))
    else:
        caps_postpgie.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12, block-linear=false"))
    '''

    if number_sources > 1:
        print("Creating tiler {}\n ".format(number_sources))
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not tiler:
            sys.stderr.write(" Unable to create tiler \n")

    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    # 0 cpu mode 1 gpu mode
    nvosd.set_property('process-mode', 1)

    nvvidconv_postosd = Gst.ElementFactory.make(
        "nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # Create a caps filter
    caps_postosd = Gst.ElementFactory.make("capsfilter", "filter_postosd")
    #caps_postosd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))
    if is_aarch64():
        caps_postosd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, block-linear=false"))
    else:
        caps_postosd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12, block-linear=false"))

    if is_aarch64():
        egltrans = Gst.ElementFactory.make("nvegltransform", "egltrans")
        if not egltrans:
            sys.stderr.write(" Unable to create egl transform")

    # Make the display sink
    sink = Gst.ElementFactory.make("nveglglessink", "nveglglessink")
    if not sink:
        sys.stderr.write(" Unable to create display")

    #streammux.set_property("width", 1920)
    streammux.set_property("width", input_width)
    #streammux.set_property("height", 1080)
    streammux.set_property("height", input_height)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", 4000000)

    print("Adding elements to Pipeline \n")
    if number_sources > 1:
        tiler_rows = int(math.sqrt(number_sources))
        tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
        #tiler.set_property("width", TILED_OUTPUT_WIDTH)
        tiler.set_property("width", tiled_output_width)
        #tiler.set_property("height", TILED_OUTPUT_HEIGHT)
        tiler.set_property("height", tiled_output_height)
        #sink.set_property("qos", 0)

    pipeline.add(nvvidconv_postmux)
    pipeline.add(cap_postmux)
    pipeline.add(pgie)
    if number_sources > 1:
        pipeline.add(tiler)
    pipeline.add(nvvidconv_postpgie)
    pipeline.add(caps_postpgie)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps_postosd)
    if is_aarch64():
        pipeline.add(egltrans)
    pipeline.add(sink)

    streammux.link(nvvidconv_postmux)
    nvvidconv_postmux.link(cap_postmux)
    cap_postmux.link(pgie)
    pgie.link(caps_postpgie)
    caps_postpgie.link(nvvidconv_postpgie)
    if number_sources > 1:
        nvvidconv_postpgie.link(tiler)
        tiler.link(nvosd)
    else:
        nvvidconv_postpgie.link(nvosd)

    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps_postosd)
    if is_aarch64():
        caps_postosd.link(egltrans)
        egltrans.link(sink)
    else:
        caps_postosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)


    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except BaseException:
        #Gst.debug_bin_to_dot_file_with_ts(pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        pass

    pipeline.set_state(Gst.State.PAUSED)
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, "pipeline")

    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='NVDSVideoTemplate Sample Application Help ')
    parser.add_argument("-i", "--input",
                    help="Path to input H264 elementry stream", nargs="+", default=["a"], required=True)
    #parser.add_argument("-c", "--codec", default="H264",
    #                help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264','H265'])
    #parser.add_argument("-b", "--bitrate", default=4000000,
    #              help="Set the encoding bitrate ", type=int)
    parser.add_argument("-w", "--width", default=1920,
                        help="Set the input width ", type=int)
    parser.add_argument("-e", "--height", default=1080,
                        help="Set the input height ", type=int)
    parser.add_argument("-tw", "--tiled-width", default=1920,
                        help="Set the tiled output width ", type=int)
    parser.add_argument("-th", "--tiled-height", default=1080,
                        help="Set the tiled output height ", type=int)
    parser.add_argument("-m", "--model-path", default="",
                        help="Set the trt model path", type=str)
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global input_width
    input_width = args.width
    global input_height
    input_height = args.height
    global tiled_output_width
    tiled_output_width = args.tiled_width
    global tiled_output_height
    tiled_output_height = args.tiled_height
    global model_path
    model_path = args.model_path
    #global codec
    #global bitrate
    global stream_path
    #global gie
    #gie = args.gie
    #codec = args.codec
    #bitrate = args.bitrate
    stream_path = args.input
    return stream_path

if __name__ == '__main__':
    parse_ds_version()
    stream_path = parse_args()
    print("Stream path ", stream_path)
    sys.exit(main(stream_path))
