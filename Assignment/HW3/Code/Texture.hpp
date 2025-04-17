//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include <Eigen/Eigen>
#include <algorithm>

#include "global.hpp"
#include "opencv2/opencv.hpp"

class Texture {
  private:
	cv::Mat image_data;

  public:
	Texture(std::string const& name) {
		image_data = cv::imread(name);
		cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
		width = image_data.cols;
		height = image_data.rows;
	}

	int width, height;

	Eigen::Vector3f getColor(float u, float v) {
		u = std::clamp(u, 0.f, 1.f); // Ensure u is in the range [0, 1]
		v = std::clamp(v, 0.f, 1.f); // Ensure v is in the range [0, 1]

		auto u_img = u * width;
		auto v_img = (1 - v) * height;

		// see https://zhuanlan.zhihu.com/p/608098962 for more issues
		if (u_img >= width - 1) u_img = width - 1;
		if (v_img >= height - 1) v_img = height - 1;

		auto color = image_data.at<cv::Vec3b>(v_img, u_img);
		return Eigen::Vector3f(color[0], color[1], color[2]);
	}

	Eigen::Vector3f getColorBilinear(float u, float v) {
		u = std::clamp(u, 0.f, 1.f); // Ensure u is in the range [0, 1]
		v = std::clamp(v, 0.f, 1.f); // Ensure v is in the range [0, 1]

		auto u_img = u * width;
		auto v_img = (1 - v) * height;

		// see https://zhuanlan.zhihu.com/p/608098962 for more issues
		if (u_img >= width - 1) u_img = width - 1;
		if (v_img >= height - 1) v_img = height - 1;

		auto x = static_cast<int>(u_img);
		auto y = static_cast<int>(v_img);

		auto x_diff = u_img - x;
		auto y_diff = v_img - y;

		cv::Vec3b color[2][2];
		color[0][0] = image_data.at<cv::Vec3b>(y, x);
		color[0][1] = image_data.at<cv::Vec3b>(y, x + 1);
		color[1][0] = image_data.at<cv::Vec3b>(y + 1, x);
		color[1][1] = image_data.at<cv::Vec3b>(y + 1, x + 1);

		auto lerp = [](float const& x, cv::Vec3b const& v0, cv::Vec3b const& v1) {
			return cv::Vec3b(v0[0] * (1 - x) + v1[0] * x, v0[1] * (1 - x) + v1[1] * x, v0[2] * (1 - x) + v1[2] * x);
		};

		auto color_u0 = lerp(x_diff, color[0][0], color[0][1]);
		auto color_u1 = lerp(x_diff, color[1][0], color[1][1]);
		auto color_final = lerp(y_diff, color_u0, color_u1);

		return Eigen::Vector3f(color_final[0], color_final[1], color_final[2]);
	}
};
#endif // RASTERIZER_TEXTURE_H
