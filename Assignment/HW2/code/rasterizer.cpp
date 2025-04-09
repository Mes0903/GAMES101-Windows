//
// Modified by [Your Name] on [Date] for Dual MSAA/SSAA with Resolve Pass
//

#include "rasterizer.hpp"
#include <math.h>
#include <algorithm>
#include <vector>
#include "opencv2/opencv.hpp"

namespace {
// Check if the point (x,y) is inside the triangle defined by v[0], v[1], v[2]
bool insideTriangle(float x, float y, Eigen::Vector3f const (&v)[3]) {
	Eigen::Vector3f px0 = v[0] - Eigen::Vector3f(x, y, 0);
	Eigen::Vector3f px1 = v[1] - Eigen::Vector3f(x, y, 0);
	Eigen::Vector3f px2 = v[2] - Eigen::Vector3f(x, y, 0);
	float c0 = px0.cross(px1).z();
	float c1 = px1.cross(px2).z();
	float c2 = px2.cross(px0).z();
	return c0 > 0 && c1 > 0 && c2 > 0;
}

// Compute barycentric coordinates for (x, y) relative to triangle vertices v
std::tuple<float, float, float> computeBarycentric2D(float x, float y, Eigen::Vector3f const (&v)[3]) {
	float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
			   (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
	float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
			   (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
	float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
			   (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
	return {c1, c2, c3};
}

// Helper to construct a 4D vector (for homogeneous coordinates)
auto to_vec4(Eigen::Vector3f const& v3, float w = 1.0f) { return Vector4f(v3.x(), v3.y(), v3.z(), w); }
} // namespace

#if defined(SSAA) || defined(MSAA)
// Map 2D sub-sample coordinates to a 1D index within the per-pixel supersample buffer.
int rst::rasterizer::get_ss_index(int sx, int sy) { return sy * MULTISAMPLE_Y + sx; }

// Resolve the supersample buffer into the final frame buffer.
// For SSAA, we average all the per-sample colors into one final pixel color;
// for MSAA we assume the final color is already set during rasterization.
#if defined(SSAA)
void rst::rasterizer::resolve_ssaa_buffer() {
	int total_samples = MULTISAMPLE_X * MULTISAMPLE_Y;
	for (size_t i = 0; i < frame_buf.size(); i++) {
		Eigen::Vector3f color_sum(0, 0, 0);
		for (int s = 0; s < total_samples; s++) { color_sum += ss_frame_buf[i][s]; }
		frame_buf[i] = color_sum / total_samples;
	}
}
#endif
#endif

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
		// Homogeneous division.
		for (auto& vec : v) { vec /= vec.w(); }
		// Viewport transformation.
		for (auto& vert : v) {
			vert.x() = 0.5 * width * (vert.x() + 1.0);
			vert.y() = 0.5 * height * (vert.y() + 1.0);
			vert.z() = vert.z() * f1 + f2;
		}

		for (int i = 0; i < 3; ++i) { t.setVertex(i, v[i].head<3>()); }

		auto col_x = col[i[0]];
		auto col_y = col[i[1]];
		auto col_z = col[i[2]];
		t.setColor(0, col_x[0], col_x[1], col_x[2]);
		t.setColor(1, col_y[0], col_y[1], col_y[2]);
		t.setColor(2, col_z[0], col_z[1], col_z[2]);

		rasterize_triangle(t);
	}

#if defined(SSAA)
	resolve_ssaa_buffer(); // Resolve the supersample buffer after rasterization.
#endif
}

// Rasterize a single triangle.
void rst::rasterizer::rasterize_triangle(Triangle const& t) {
	auto v = t.toVector4();

	// Determine bounding box for the triangle.
	int min_x = std::floor(std::min({v[0].x(), v[1].x(), v[2].x()}));
	int min_y = std::floor(std::min({v[0].y(), v[1].y(), v[2].y()}));
	int max_x = std::ceil(std::max({v[0].x(), v[1].x(), v[2].x()}));
	int max_y = std::ceil(std::max({v[0].y(), v[1].y(), v[2].y()}));

	for (int x = min_x; x <= max_x; ++x) {
		for (int y = min_y; y <= max_y; ++y) {

#if defined(SSAA)
			/*
			 * For black edge problem, check the following link
			 * https://zhuanlan.zhihu.com/p/454001952
			 * https://blog.csdn.net/ycrsw/article/details/123910834
			 * https://games-cn.org/forums/topic/%E3%80%90%E6%80%BB%E7%BB%93%E3%80%91msaa%E4%B8%AD%E9%BB%91%E7%BA%BF%E9%97%AE%E9%A2%98%E7%9A%84%E5%87%BA%E7%8E%B0%E5%8E%9F%E5%9B%A0%E4%BB%A5%E5%8F%8A%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/
			 */
			int pixel_index = get_index(x, y);
			for (int sx = 0; sx < MULTISAMPLE_X; ++sx) {
				for (int sy = 0; sy < MULTISAMPLE_Y; ++sy) {
					float sampleX = x + sx * (1 / (float)MULTISAMPLE_X) + (1 / 2 * (float)MULTISAMPLE_X);
					float sampleY = y + sy * (1 / (float)MULTISAMPLE_Y) + (1 / 2 * (float)MULTISAMPLE_Y);
					if (insideTriangle(sampleX, sampleY, t.v)) {
						auto [alpha, beta, gamma] = computeBarycentric2D(sampleX, sampleY, t.v);
						float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
						float z_interpolated = (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w()) * w_reciprocal;

						int sample_index = get_ss_index(sx, sy);
						if (z_interpolated < ss_depth_buf[pixel_index][sample_index]) {
							ss_depth_buf[pixel_index][sample_index] = z_interpolated;
							ss_frame_buf[pixel_index][sample_index] = t.getColor();
						}
					}
				}
			}
#elif defined(MSAA)
			if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
				int cnt = 0;
				int pixel_index = get_index(x, y);
				float min_depth = std::numeric_limits<float>::infinity();
				for (int sx = 0; sx < MULTISAMPLE_X; ++sx) {
					for (int sy = 0; sy < MULTISAMPLE_Y; ++sy) {
						float sampleX = x + sx * (1 / (float)MULTISAMPLE_X) + (1 / 2 * (float)MULTISAMPLE_X);
						float sampleY = y + sy * (1 / (float)MULTISAMPLE_Y) + (1 / 2 * (float)MULTISAMPLE_Y);
						if (insideTriangle(sampleX, sampleY, t.v)) {
							auto [alpha, beta, gamma] = computeBarycentric2D(sampleX, sampleY, t.v);
							float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
							float z_interpolated = (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w()) * w_reciprocal;

							int sample_index = get_ss_index(sx, sy);
							if (z_interpolated < ss_depth_buf[pixel_index][sample_index]) {
								cnt++;
								ss_depth_buf[pixel_index][sample_index] = z_interpolated;
								min_depth = std::min(min_depth, z_interpolated); // Store the minimum depth as the final depth for depth buffer.
							}
						}
					}
				}

				if (min_depth < depth_buf[pixel_index]) {
					depth_buf[pixel_index] = min_depth;
					frame_buf[pixel_index] = t.getColor() * (float)cnt / (MULTISAMPLE_X * MULTISAMPLE_Y);
				}
			}
#else
			// Regular single-sample rasterization.
			if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
				auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5, y + 0.5, t.v);
				float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
				float z_interpolated = (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w()) * w_reciprocal;

				int pixel_index = get_index(x, y);
				if (z_interpolated < depth_buf[pixel_index]) {
					depth_buf[pixel_index] = z_interpolated;
					frame_buf[pixel_index] = t.getColor();
				}
			}
#endif
		}
	}
}

void rst::rasterizer::set_model(Eigen::Matrix4f const& m) { model = m; }
void rst::rasterizer::set_view(Eigen::Matrix4f const& v) { view = v; }
void rst::rasterizer::set_projection(Eigen::Matrix4f const& p) { projection = p; }

void rst::rasterizer::clear(rst::Buffers buff) {
	if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
		std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
#if defined(SSAA)
		for (auto& arr : ss_frame_buf) { std::fill(arr.begin(), arr.end(), Eigen::Vector3f{0, 0, 0}); }
#endif
	}

	if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
		std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
#if defined(SSAA) || defined(MSAA)
		for (auto& arr : ss_depth_buf) { std::fill(arr.begin(), arr.end(), std::numeric_limits<float>::infinity()); }
#endif
	}
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
	frame_buf.resize(w * h);
	depth_buf.resize(w * h);
#if defined(SSAA) || defined(MSAA)
	ss_depth_buf.resize(w * h);
#if defined(SSAA)
	ss_frame_buf.resize(w * h);
#endif
#endif
}

int rst::rasterizer::get_index(int x, int y) { return (height - 1 - y) * width + x; }

void rst::rasterizer::set_pixel(Eigen::Vector3f const& point, Eigen::Vector3f const& color) {
	int ind = (height - 1 - point.y()) * width + point.x();
	frame_buf[ind] = color;
}
