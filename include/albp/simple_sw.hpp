#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>



namespace albp {

template<class T>
struct Matrix2D {
  std::vector<T> data;
  size_t height;
  size_t width;

  Matrix2D() = default;
  Matrix2D(int height, int width) : height(height), width(width) {
    if(height>width)
      throw std::runtime_error("width must be >= height!");
    data.resize(width*height);
  }

  /**
    @brief Creates a raster from a nested initializer list

    @param[in] values The raster

    @example
      Array2D<int> x = {{1,2,3},{4,5,6}}
  */
  Matrix2D(std::initializer_list<std::initializer_list<T>> values) {
    height = values.size();
    width  = values.begin()->size();

    if(height>width)
      throw std::runtime_error("width must be >= height!");

    for(const auto &x: values){
      if(x.size()!=width)
        throw std::runtime_error("All rows of the array must be the same width!");
    }

    data.resize(width*height);

    size_t i=0;
    size_t j=0;
    for(const auto &row: values){
      j=0;
      for(const auto &col: row){
        operator()(i,j) = col;
        j++;
      }
      i++;
    }
    (void)height; //Suppress unused variable warning with NDEBUG
    assert(i==height);
  }

  T&       operator()(const size_t i, const size_t j)       { assert(i<height); assert(j<width); return data[i*width+j]; }
  const T& operator()(const size_t i, const size_t j) const { assert(i<height); assert(j<width); return data[i*width+j]; }
  size_t   size() const { return height*width; }
  T max() const { return *std::max_element(data.begin(), data.end()); }
};



struct SimpleSmithWatermanResult {
  std::string seqa;
  std::string seqb;
  Matrix2D<int> E;
  Matrix2D<int> F;
  Matrix2D<int> H;
  int score;
  int size;
};



template<class T>
void print_matrix(
  const std::string message,
  const std::string &seqa,
  const std::string &seqb,
  const Matrix2D<T> &matrix
){
  std::cerr<<"\n"<<message<<":\n";

  std::cerr<<"  ";
  for(const auto &x: seqb){
    std::cerr<<std::setw(2)<<x<<" ";
  }
  std::cerr<<"\n";

  for(size_t i=0;i<seqa.size();i++){
    std::cerr<<seqa[i]<<" ";
    for(size_t j=0;j<seqb.size();j++){
      std::cerr<<std::setw(2)<<(int)matrix(i,j)<<" ";
    }
    std::cerr<<"\n";
  }
}



template<class T>
std::vector<T> extract_diagonal(const Matrix2D<T> &matrix, const int diagonal_i){
  std::vector<T> diagonal;
  diagonal.reserve(matrix.height);

  if(diagonal_i<0){
    return {};
  }

  if((size_t)diagonal_i>=matrix.width+matrix.height-1){
    throw std::runtime_error("diagonal_i was too large for the matrix!");
  }

  int i = diagonal_i;
  int j = 0;
  if((size_t)diagonal_i>=matrix.height){
    i = matrix.height-1;
    j = diagonal_i - matrix.height + 1;
  }

  for(;i>=0 && j<(int)matrix.width;i--,j++){
    diagonal.push_back(matrix(i,j));
  }
  return diagonal;
}



SimpleSmithWatermanResult simple_smith_waterman(
  const std::string &seqa,
  const std::string &seqb,
  const int gap_open,
  const int gap_extend,
  const int match_score,
  const int mismatch_score
);

}