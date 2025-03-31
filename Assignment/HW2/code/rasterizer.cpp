//
// Created by goksu on 4/6/19.
//

#include "rasterizer.hpp"
#include <math.h>
#include <algorithm>
#include <vector>
#include "opencv2/opencv.hpp"

namespace {
bool insideTriangle(int x, int y, Eigen::Vector3f const (&v)[3]) {
	// TODO : Implement this function to check if the point (x, y) is inside the triangle represented by v[0], v[1], v[2]
	Eigen::Vector3f px0 = v[0] - Eigen::Vector3f(x, y, 0);
	Eigen::Vector3f px1 = v[1] - Eigen::Vector3f(x, y, 0);
	Eigen::Vector3f px2 = v[2] - Eigen::Vector3f(x, y, 0);
	float c0 = px0.cross(px1).z();
	float c1 = px1.cross(px2).z();
	float c2 = px2.cross(px0).z();

	return c0 > 0 && c1 > 0 && c2 > 0;
}
} // namespace

rst::pos_buf_id rst::rasterizer::load_positions(std::vector<Eigen::Vector3f> const& positions) {
	auto id = get_next_id();
	pos_buf.emplace(id, positions);

	return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(std::vector<Eigen::Vector3i> const& indices) {
	auto id = get_next_id();
	ind_buf.emplace(id, indices);

	return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(std::vector<Eigen::Vector3f> const& cols) {
	auto id = get_next_id();
	col_buf.emplace(id, cols);

	return {id};
}

auto to_vec4(Eigen::Vector3f const& v3, float w = 1.0f) { return Vector4f(v3.x(), v3.y(), v3.z(), w); }

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, Vector3f const (&v)[3]) {
	float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
			   (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
	float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
			   (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
	float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
			   (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
	return {c1, c2, c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type) {
	auto& buf = pos_buf[pos_buffer.pos_id];
	auto& ind = ind_buf[ind_buffer.ind_id];
	auto& col = col_buf[col_buffer.col_id];

	float f1 = (50 - 0.1) / 2.0;
	float f2 = (50 + 0.1) / 2.0;

	Eigen::Matrix4f mvp = projection * view * model;
	for (auto& i : ind) {
		Triangle t;
		Eigen::Vector4f v[] = {mvp * to_vec4(buf[i[0]], 1.0f), mvp * to_vec4(buf[i[1]], 1.0f), mvp * to_vec4(buf[i[2]], 1.0f)};
		// Homogeneous division
		for (auto& vec : v) { vec /= vec.w(); }
		// Viewport transformation
		for (auto& vert : v) {
			vert.x() = 0.5 * width * (vert.x() + 1.0);
			vert.y() = 0.5 * height * (vert.y() + 1.0);
			vert.z() = vert.z() * f1 + f2;
		}

		for (int i = 0; i < 3; ++i) {
			t.setVertex(i, v[i].head<3>());
			t.setVertex(i, v[i].head<3>());
			t.setVertex(i, v[i].head<3>());
		}

		auto col_x = col[i[0]];
		auto col_y = col[i[1]];
		auto col_z = col[i[2]];

		t.setColor(0, col_x[0], col_x[1], col_x[2]);
		t.setColor(1, col_y[0], col_y[1], col_y[2]);
		t.setColor(2, col_z[0], col_z[1], col_z[2]);

		rasterize_triangle(t);
	}
}

// Screen space rasterization
void rst::rasterizer::rasterize_triangle(Triangle const& t) {
	auto v = t.toVector4();

	// TODO : Find out the bounding box of current triangle.
	// iterate through the pixel and find if the current ixel is inside the triangle

	int min_x = std::floor(std::min({v[0].x(), v[1].x(), v[2].x()}));
	int min_y = std::floor(std::min({v[0].y(), v[1].y(), v[2].y()}));
	int max_x = std::ceil(std::max({v[0].x(), v[1].x(), v[2].x()}));
	int max_y = std::ceil(std::max({v[0].y(), v[1].y(), v[2].y()}));

	for (int x = min_x; x < max_x + 1; ++x) {
		for (int y = min_y; y < max_y + 1; ++y) {
			if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
				// If so, use the following code to get the interpolated z value.
				auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
				float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
				float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
				z_interpolated *= w_reciprocal;

				if (z_interpolated < depth_buf[get_index(x, y)]) {
					depth_buf[get_index(x, y)] = z_interpolated;
					frame_buf[get_index(x, y)] = t.getColor();
				}
			}
		}
	}
}

void rst::rasterizer::set_model(Eigen::Matrix4f const& m) { model = m; }

void rst::rasterizer::set_view(Eigen::Matrix4f const& v) { view = v; }

void rst::rasterizer::set_projection(Eigen::Matrix4f const& p) { projection = p; }

void rst::rasterizer::clear(rst::Buffers buff) {
	if ((buff & rst::Buffers::Color) == rst::Buffers::Color) { std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0}); }
	if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) { std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity()); }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
	frame_buf.resize(w * h);
	depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y) { return (height - 1 - y) * width + x; }

void rst::rasterizer::set_pixel(Eigen::Vector3f const& point, Eigen::Vector3f const& color) {
	// old index: auto ind = point.y() + point.x() * width;
	auto ind = (height - 1 - point.y()) * width + point.x();
	frame_buf[ind] = color;
}