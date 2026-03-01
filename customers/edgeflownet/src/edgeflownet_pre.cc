#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

struct edgeflownet_pre_properties {
    int width = 0;
    int height = 0;
    float scale = 0.003921568859f; // 1/255
    float mean0 = 0.0f;
    float mean1 = 0.0f;
    float mean2 = 0.0f;
    float std0 = 1.0f;
    float std1 = 1.0f;
    float std2 = 1.0f;
    float quant_scale = 0.003921568859f;
    int quant_zeropoint = -128;
    bool swap_rb = false;
    bool has_prev = false;
    cv::Mat prev;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
    static const std::unordered_set<std::string> allowed_properties{
        "width",
        "height",
        "scale",
        "mean0",
        "mean1",
        "mean2",
        "std0",
        "std1",
        "std2",
        "quant_scale",
        "quant_zeropoint",
        "swap_rb",
    };
    return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &)
{
    std::shared_ptr<edgeflownet_pre_properties> prop =
        std::make_shared<edgeflownet_pre_properties>();
    prop->width = Ax::get_property(input, "width", "edgeflownet_pre_properties", prop->width);
    prop->height = Ax::get_property(input, "height", "edgeflownet_pre_properties", prop->height);
    prop->scale = Ax::get_property(input, "scale", "edgeflownet_pre_properties", prop->scale);
    prop->mean0 = Ax::get_property(input, "mean0", "edgeflownet_pre_properties", prop->mean0);
    prop->mean1 = Ax::get_property(input, "mean1", "edgeflownet_pre_properties", prop->mean1);
    prop->mean2 = Ax::get_property(input, "mean2", "edgeflownet_pre_properties", prop->mean2);
    prop->std0 = Ax::get_property(input, "std0", "edgeflownet_pre_properties", prop->std0);
    prop->std1 = Ax::get_property(input, "std1", "edgeflownet_pre_properties", prop->std1);
    prop->std2 = Ax::get_property(input, "std2", "edgeflownet_pre_properties", prop->std2);
    prop->quant_scale =
        Ax::get_property(input, "quant_scale", "edgeflownet_pre_properties", prop->quant_scale);
    prop->quant_zeropoint = Ax::get_property(
        input, "quant_zeropoint", "edgeflownet_pre_properties", prop->quant_zeropoint);
    prop->swap_rb = Ax::get_property(input, "swap_rb", "edgeflownet_pre_properties", prop->swap_rb);
    return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const edgeflownet_pre_properties *prop, Ax::Logger &)
{
    if (!std::holds_alternative<AxVideoInterface>(interface)) {
        throw std::runtime_error("edgeflownet_pre works on video input only");
    }
    auto &info = std::get<AxVideoInterface>(interface).info;
    const int out_h = (prop->height > 0) ? prop->height : info.height;
    const int out_w = (prop->width > 0) ? prop->width : info.width;

    AxDataInterface output = AxTensorsInterface(1);
    auto &tensor = std::get<AxTensorsInterface>(output)[0];
    tensor.bytes = 1;
    tensor.sizes = std::vector<int>{ 1, out_h, out_w, 6 };
    return output;
}

static inline int8_t quantize_val(float v, float scale, int zero_point)
{
    int q = static_cast<int>(std::nearbyint(v / scale + static_cast<float>(zero_point)));
    q = std::min(127, std::max(-128, q));
    return static_cast<int8_t>(q);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const edgeflownet_pre_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &)
{
    cv::ocl::setUseOpenCL(false);

    if (!std::holds_alternative<AxVideoInterface>(input)) {
        throw std::runtime_error("edgeflownet_pre expects video input");
    }
    auto &input_video = std::get<AxVideoInterface>(input);
    const int in_w = input_video.info.width;
    const int in_h = input_video.info.height;
    const auto format = input_video.info.format;
    const int channels = AxVideoFormatNumChannels(format);
    if (channels <= 0) {
        throw std::runtime_error("edgeflownet_pre expects image input");
    }

    cv::Mat input_mat(cv::Size(in_w, in_h),
        Ax::opencv_type_u8(format),
        input_video.data, input_video.info.stride);

    cv::Mat rgb_mat;
    switch (format) {
    case AxVideoFormat::RGB:
        rgb_mat = input_mat;
        break;
    case AxVideoFormat::BGR:
        cv::cvtColor(input_mat, rgb_mat, cv::COLOR_BGR2RGB);
        break;
    case AxVideoFormat::RGBA:
    case AxVideoFormat::RGBx:
        cv::cvtColor(input_mat, rgb_mat, cv::COLOR_RGBA2RGB);
        break;
    case AxVideoFormat::BGRA:
    case AxVideoFormat::BGRx:
        cv::cvtColor(input_mat, rgb_mat, cv::COLOR_BGRA2RGB);
        break;
    case AxVideoFormat::GRAY8:
        cv::cvtColor(input_mat, rgb_mat, cv::COLOR_GRAY2RGB);
        break;
    default:
        throw std::runtime_error("edgeflownet_pre expects RGB/BGR/RGBA/BGRA/GRAY8 input");
    }

    const int out_w = (prop->width > 0) ? prop->width : in_w;
    const int out_h = (prop->height > 0) ? prop->height : in_h;

    cv::Mat resized;
    if (out_w != in_w || out_h != in_h) {
        cv::resize(rgb_mat, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = rgb_mat;
    }

    cv::Mat curr_float;
    resized.convertTo(curr_float, CV_32FC3, prop->scale);

    auto *mutable_prop = const_cast<edgeflownet_pre_properties *>(prop);
    cv::Mat prev_float;
    if (!mutable_prop->has_prev) {
        prev_float = curr_float;
        mutable_prop->prev = curr_float.clone();
        mutable_prop->has_prev = true;
    } else {
        prev_float = mutable_prop->prev;
        mutable_prop->prev = curr_float.clone();
    }

    auto &output_tensor = std::get<AxTensorsInterface>(output)[0];
    int8_t *out = static_cast<int8_t *>(output_tensor.data);
    const int hw = out_h * out_w;
    const int out_c = 6;
    std::fill(out, out + hw * out_c, static_cast<int8_t>(prop->quant_zeropoint));
    const float mean[3] = { prop->mean0, prop->mean1, prop->mean2 };
    const float stdv[3] = { prop->std0, prop->std1, prop->std2 };

    const int idx0 = prop->swap_rb ? 2 : 0;
    const int idx2 = prop->swap_rb ? 0 : 2;
    const int map[3] = { idx0, 1, idx2 };

    for (int y = 0; y < out_h; ++y) {
        const cv::Vec3f *p_row = prev_float.ptr<cv::Vec3f>(y);
        const cv::Vec3f *c_row = curr_float.ptr<cv::Vec3f>(y);
        for (int x = 0; x < out_w; ++x) {
            const int base = (y * out_w + x) * out_c;
            for (int c = 0; c < 3; ++c) {
                const int cc = map[c];
                float pv = (p_row[x][cc] - mean[c]) / stdv[c];
                float cv = (c_row[x][cc] - mean[c]) / stdv[c];
                out[base + c] =
                    quantize_val(pv, prop->quant_scale, prop->quant_zeropoint);
                out[base + 3 + c] =
                    quantize_val(cv, prop->quant_scale, prop->quant_zeropoint);
            }
        }
    }
}
