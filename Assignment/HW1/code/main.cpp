#include <cmath>
#include <iostream>

#include "Eigen/Eigen"
#include "Triangle.hpp"
#include "opencv2/opencv.hpp"
#include "rasterizer.hpp"

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
	Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

	Eigen::Matrix4f translate;
	translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1, -eye_pos[2], 0, 0, 0, 1;

	view = translate * view;

	return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle) {
	Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

	// TODO: Implement this function
	// Create the model matrix for rotating the triangle around the Z axis.
	// Then return it.
	rotation_angle = rotation_angle / 180.0 * MY_PI;
	model(0, 0) = std::cos(rotation_angle);
	model(0, 1) = -std::sin(rotation_angle);
	model(1, 0) = std::sin(rotation_angle);
	model(1, 1) = std::cos(rotation_angle);

	return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar) {
	/* clang-format off */
	Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

	// Since the OpenGL coordinate system is right-handed, we need to flip the z axis.
	// See more: https://zhuanlan.zhihu.com/p/509902950
	zNear = -zNear;
	zFar = -zFar;

	const float n = zNear;
	const float f = zFar;

	// TODO: Implement this function
	// Create the projection matrix for the given parameters.
	// Then return it.
	eye_fov = eye_fov / 180.0 * MY_PI;
	float t = std::tan(eye_fov / 2) * std::abs(zNear);
	float b = -t;
	float r = t * aspect_ratio;
	float l = -r;

	// persp->ortho
	Eigen::Matrix4f Mpo;
	Mpo << n, 0, 0, 0,
			   0, n, 0, 0,
				 0, 0, n + f, -n * f,
				 0, 0, 1, 0;

	// ortho
	Eigen::Matrix4f Mo1, Mo2;
	Mo1 << 2 / (r - l), 0, 0, 0,
				 0, 2 / (t - b), 0, 0, 
				 0, 0, 2 / (n - f), 0,
				 0, 0, 0, 1;
	Mo2 << 1, 0, 0, -(r + l) / 2,
				 0, 1, 0, -(t + b) / 2,
				 0, 0, 1, -(n + f) / 2,
				 0, 0, 0, 1;

	// persp
	projection = Mo1 * Mo2 * Mpo;

	// or directory set the projection matrix
	// Eigen::Matrix4f projection2;
	// projection2 << 2 * n / (r - l), 0, -(r + l) / (r - l), 0,
	// 							 0, 2 * n / (t - b), -(t + b) / (t - b), 0,
	// 							 0, 0, (n + f) / (n - f), -2 * n * f / (n - f),
	// 							 0, 0, 1, 0;
	// projection = projection2;

	Eigen::Matrix4f FlipZ = Eigen::Matrix4f::Identity();
	FlipZ(2, 2) = -1;
	projection = FlipZ * projection;

	return projection;
	/* clang-format on */
}

int main(int argc, char const** argv) {
	float angle = 0;
	bool command_line = false;
	std::string filename = "output.png";

	if (argc >= 3) {
		command_line = true;
		angle = std::stof(argv[2]); // -r by default
		if (argc == 4) {
			filename = std::string(argv[3]);
		} else
			return 0;
	}

	rst::rasterizer r(700, 700);
	Eigen::Vector3f eye_pos = {0, 0, 5};								   // camera position
	std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}}; // triangle position
	std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

	auto pos_id = r.load_positions(pos);
	auto ind_id = r.load_indices(ind);

	int key = 0;
	int frame_count = 0;

	if (command_line) {
		r.clear(rst::Buffers::Color | rst::Buffers::Depth);

		r.set_model(get_model_matrix(angle));
		r.set_view(get_view_matrix(eye_pos));
		r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

		r.draw(pos_id, ind_id, rst::Primitive::Triangle);
		cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
		image.convertTo(image, CV_8UC3, 1.0f);

		cv::imwrite(filename, image);

		return 0;
	}

	while (key != 27) {
		r.clear(rst::Buffers::Color | rst::Buffers::Depth);

		r.set_model(get_model_matrix(angle));
		r.set_view(get_view_matrix(eye_pos));
		r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

		r.draw(pos_id, ind_id, rst::Primitive::Triangle);

		cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
		image.convertTo(image, CV_8UC3, 1.0f);
		cv::imshow("image", image);
		key = cv::waitKey(10);

		std::cout << "frame count: " << frame_count++ << '\n';

		if (key == 'a') {
			angle += 10;
		} else if (key == 'd') {
			angle -= 10;
		}
	}

	return 0;
}
