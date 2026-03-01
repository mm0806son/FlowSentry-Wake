#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"
#include "edgeflownet_meta.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>

struct edgeflownet_post_properties {
    float max_flow = 50.0f;
    float gamma = 1.0f;
    int out_width = 0;
    int out_height = 0;
    std::string meta_key = "edgeflownet_flow";
    std::string master_meta{};
};

struct tensor_view {
    int index = 0;
    int n = 0;
    int h = 0;
    int w = 0;
    int c = 0;
    bool nhwc = true;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
    static const std::unordered_set<std::string> allowed_properties{
        "max_flow",
        "gamma",
        "out_width",
        "out_height",
        "meta_key",
        "master_meta",
    };
    return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &)
{
    auto prop = std::make_shared<edgeflownet_post_properties>();
    prop->max_flow = Ax::get_property(input, "max_flow", "edgeflownet_post_properties",
        prop->max_flow);
    prop->gamma = Ax::get_property(input, "gamma", "edgeflownet_post_properties",
        prop->gamma);
    prop->out_width = Ax::get_property(
        input, "out_width", "edgeflownet_post_properties", prop->out_width);
    prop->out_height = Ax::get_property(
        input, "out_height", "edgeflownet_post_properties", prop->out_height);
    prop->meta_key = Ax::get_property(
        input, "meta_key", "edgeflownet_post_properties", prop->meta_key);
    prop->master_meta = Ax::get_property(
        input, "master_meta", "edgeflownet_post_properties", prop->master_meta);
    return prop;
}

static bool parse_tensor_view(const AxTensorInterface &tensor, tensor_view &view)
{
    if (tensor.sizes.size() != 4) {
        return false;
    }
    const int n = tensor.sizes[0];
    const int s1 = tensor.sizes[1];
    const int s2 = tensor.sizes[2];
    const int s3 = tensor.sizes[3];

    auto is_spatial = [](int v) { return v >= 8; };
    const bool nhwc_ok = is_spatial(s1) && is_spatial(s2);
    const bool nchw_ok = is_spatial(s2) && is_spatial(s3);

    if (s3 <= 4 && nhwc_ok) {
        view = { view.index, n, s1, s2, s3, true };
        return true;
    }
    if (s1 <= 4 && nchw_ok) {
        view = { view.index, n, s2, s3, s1, false };
        return true;
    }
    if (nhwc_ok) {
        view = { view.index, n, s1, s2, s3, true };
        return true;
    }
    if (nchw_ok) {
        view = { view.index, n, s2, s3, s1, false };
        return true;
    }
    return false;
}

static tensor_view pick_flow_tensor(const AxTensorsInterface &tensors)
{
    tensor_view best{};
    bool found = false;
    int best_area = -1;

    for (size_t i = 0; i < tensors.size(); ++i) {
        tensor_view view{};
        view.index = static_cast<int>(i);
        if (!parse_tensor_view(tensors[i], view)) {
            continue;
        }
        if (view.c < 2) {
            continue;
        }
        const int area = view.h * view.w;
        if (!found || area > best_area) {
            best = view;
            best_area = area;
            found = true;
        }
    }

    if (!found) {
        throw std::runtime_error("edgeflownet_post: no suitable flow tensor found");
    }
    return best;
}

static cv::Mat flow_to_color(const cv::Mat &flow, float max_flow, float gamma)
{
    std::vector<cv::Mat> channels(2);
    cv::split(flow, channels);

    cv::Mat magnitude, angle;
    cv::cartToPolar(channels[0], channels[1], magnitude, angle, true);

    double max_val = 0.0;
    if (max_flow <= 0.0f) {
        cv::minMaxLoc(magnitude, nullptr, &max_val);
        if (max_val <= 0.0) {
            max_val = 1.0;
        }
        max_flow = static_cast<float>(max_val);
    }
    if (gamma <= 0.0f) {
        gamma = 1.0f;
    }

    cv::Mat hue = angle * 0.5f;
    cv::Mat norm = magnitude * (1.0f / max_flow);
    cv::threshold(norm, norm, 1.0, 1.0, cv::THRESH_TRUNC);
    if (gamma != 1.0f) {
        cv::pow(norm, gamma, norm);
    }
    cv::Mat value = norm * 255.0f;

    cv::Mat hue_u8, sat_u8, val_u8;
    hue.convertTo(hue_u8, CV_8U);
    value.convertTo(val_u8, CV_8U);
    sat_u8 = cv::Mat(hue_u8.size(), CV_8U, cv::Scalar(255));

    cv::Mat hsv, rgb;
    cv::merge(std::vector<cv::Mat>{ hue_u8, sat_u8, val_u8 }, hsv);
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
    return rgb;
}

extern "C" void
decode_to_meta(const AxTensorsInterface &tensors, const edgeflownet_post_properties *prop,
    unsigned int current_frame, unsigned int total_frames,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &, Ax::Logger &)
{
    cv::ocl::setUseOpenCL(false);

    if (total_frames <= current_frame) {
        throw std::runtime_error("edgeflownet_post: current frame out of bounds");
    }
    if (tensors.empty()) {
        throw std::runtime_error("edgeflownet_post: empty tensor list");
    }

    const auto view = pick_flow_tensor(tensors);
    const auto &tensor = tensors[view.index];

    if (tensor.bytes != 4) {
        throw std::runtime_error("edgeflownet_post expects float32 tensors");
    }
    if (!tensor.data) {
        throw std::runtime_error("edgeflownet_post: tensor data is null");
    }

    cv::Mat flow(view.h, view.w, CV_32FC2);
    const float *data = static_cast<const float *>(tensor.data);

    if (view.nhwc && view.c == 2) {
        flow = cv::Mat(view.h, view.w, CV_32FC2, const_cast<float *>(data));
    } else if (view.nhwc) {
        for (int y = 0; y < view.h; ++y) {
            for (int x = 0; x < view.w; ++x) {
                const int base = (y * view.w + x) * view.c;
                flow.at<cv::Vec2f>(y, x) = { data[base], data[base + 1] };
            }
        }
    } else {
        const int hw = view.h * view.w;
        const float *u = data;
        const float *v = data + hw;
        for (int y = 0; y < view.h; ++y) {
            for (int x = 0; x < view.w; ++x) {
                const int idx = y * view.w + x;
                flow.at<cv::Vec2f>(y, x) = { u[idx], v[idx] };
            }
        }
    }

    cv::Mat rgb = flow_to_color(flow, prop->max_flow, prop->gamma);
    const int out_w = prop->out_width > 0 ? prop->out_width : rgb.cols;
    const int out_h = prop->out_height > 0 ? prop->out_height : rgb.rows;
    if (out_w != rgb.cols || out_h != rgb.rows) {
        cv::resize(rgb, rgb, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
    }

    std::vector<uint8_t> data_u8(rgb.total() * rgb.elemSize());
    if (rgb.isContinuous()) {
        std::memcpy(data_u8.data(), rgb.data, data_u8.size());
    } else {
        const int row_bytes = rgb.cols * rgb.elemSize();
        for (int y = 0; y < rgb.rows; ++y) {
            std::memcpy(data_u8.data() + y * row_bytes, rgb.ptr(y), row_bytes);
        }
    }

    ax_utils::insert_meta<AxMetaFlowImage>(map, prop->meta_key, prop->master_meta,
        current_frame, total_frames, std::move(data_u8), out_w, out_h, 3);
}
