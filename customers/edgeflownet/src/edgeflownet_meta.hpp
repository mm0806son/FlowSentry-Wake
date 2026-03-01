// Copyright Axelera AI, 2025
#pragma once

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

#include <opencv2/imgproc.hpp>

class AxMetaFlowImage : public AxMetaBase
{
  public:
  AxMetaFlowImage(std::vector<uint8_t> data, int width, int height, int channels)
      : data_(std::move(data)), width_(width), height_(height), channels_(channels)
  {
    if (width_ <= 0 || height_ <= 0 || channels_ <= 0) {
      throw std::runtime_error("AxMetaFlowImage: invalid dimensions");
    }
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &) override
  {
    try {
      if (!video.data || channels_ != 3) {
        return;
      }
      if (video.info.width <= 0 || video.info.height <= 0 || video.info.stride <= 0) {
        return;
      }

      int dst_channels = 0;
      switch (video.info.format) {
        case AxVideoFormat::RGB:
        case AxVideoFormat::BGR:
          dst_channels = 3;
          break;
        case AxVideoFormat::RGBA:
        case AxVideoFormat::RGBx:
        case AxVideoFormat::BGRA:
        case AxVideoFormat::BGRx:
          dst_channels = 4;
          break;
        case AxVideoFormat::GRAY8:
          dst_channels = 1;
          break;
        default:
          return;
      }
      if (video.info.stride < video.info.width * dst_channels) {
        return;
      }

      cv::Mat src(height_, width_, CV_MAKETYPE(CV_8U, channels_),
          const_cast<uint8_t *>(data_.data()));
      cv::Mat dst(cv::Size(video.info.width, video.info.height),
          Ax::opencv_type_u8(video.info.format), video.data, video.info.stride);

      cv::Mat resized;
      const cv::Mat &img = (src.size() == dst.size())
          ? src
          : (cv::resize(src, resized, dst.size(), 0, 0, cv::INTER_LINEAR), resized);

      switch (video.info.format) {
        case AxVideoFormat::RGB:
          img.copyTo(dst);
          break;
        case AxVideoFormat::BGR:
          cv::cvtColor(img, dst, cv::COLOR_RGB2BGR);
          break;
        case AxVideoFormat::RGBA:
        case AxVideoFormat::RGBx:
          cv::cvtColor(img, dst, cv::COLOR_RGB2RGBA);
          break;
        case AxVideoFormat::BGRA:
        case AxVideoFormat::BGRx:
          cv::cvtColor(img, dst, cv::COLOR_RGB2BGRA);
          break;
        case AxVideoFormat::GRAY8:
          cv::cvtColor(img, dst, cv::COLOR_RGB2GRAY);
          break;
        default:
          break;
      }
    } catch (...) {
      return;
    }
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *class_meta = "FlowImage";
    auto results = std::vector<extern_meta>();
    results.push_back({ class_meta, "data", int(data_.size() * sizeof(uint8_t)),
        reinterpret_cast<const char *>(data_.data()) });
    results.push_back({ class_meta, "width", static_cast<int>(sizeof(int)),
        reinterpret_cast<const char *>(&width_) });
    results.push_back({ class_meta, "height", static_cast<int>(sizeof(int)),
        reinterpret_cast<const char *>(&height_) });
    results.push_back({ class_meta, "channels", static_cast<int>(sizeof(int)),
        reinterpret_cast<const char *>(&channels_) });
    return results;
  }

  private:
  std::vector<uint8_t> data_;
  int width_;
  int height_;
  int channels_;
};
