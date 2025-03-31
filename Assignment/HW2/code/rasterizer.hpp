//
// Created by goksu on 4/6/19.
//

#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include "Triangle.hpp"
#include "global.hpp"

using namespace Eigen;

namespace rst {
enum class Buffers { Color = 1, Depth = 2 };

inline Buffers operator|(Buffers a, Buffers b) { return Buffers((int)a | (int)b); }

inline Buffers operator&(Buffers a, Buffers b) { return Buffers((int)a & (int)b); }

enum class Primitive { Line, Triangle };

/*
 * For the curious : The draw function takes two buffer id's as its arguments. These two structs
 * make sure that if you mix up with their orders, the compiler won't compile it.
 * Aka : Type safety
 * */
struct pos_buf_id {
	int pos_id = 0;
};

struct ind_buf_id {
	int ind_id = 0;
};

struct col_buf_id {
	int col_id = 0;
};

class rasterizer {
  public:
	rasterizer(int w, int h);
	pos_buf_id load_positions(std::vector<Eigen::Vector3f> const& positions);
	ind_buf_id load_indices(std::vector<Eigen::Vector3i> const& indices);
	col_buf_id load_colors(std::vector<Eigen::Vector3f> const& colors);

	void set_model(Eigen::Matrix4f const& m);
	void set_view(Eigen::Matrix4f const& v);
	void set_projection(Eigen::Matrix4f const& p);

	void set_pixel(Eigen::Vector3f const& point, Eigen::Vector3f const& color);

	void clear(Buffers buff);

	void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type);

	std::vector<Eigen::Vector3f>& frame_buffer() { return frame_buf; }

  private:
	void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end);

	void rasterize_triangle(Triangle const& t);

	// VERTEX SHADER -> MVP -> Clipping -> /.W -> VIEWPORT -> DRAWLINE/DRAWTRI -> FRAGSHADER

  private:
	Eigen::Matrix4f model;
	Eigen::Matrix4f view;
	Eigen::Matrix4f projection;

	std::map<int, std::vector<Eigen::Vector3f>> pos_buf;
	std::map<int, std::vector<Eigen::Vector3i>> ind_buf;
	std::map<int, std::vector<Eigen::Vector3f>> col_buf;

	std::vector<Eigen::Vector3f> frame_buf;

	std::vector<float> depth_buf;
	int get_index(int x, int y);

	int width, height;

	int next_id = 0;
	int get_next_id() { return next_id++; }
};
} // namespace rst
