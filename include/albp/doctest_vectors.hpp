#pragma once

#include <albp/doctest.hpp>

#include <thrust/device_vector.h>

#include <sstream>
#include <vector>

namespace doctest {
  template <typename T>
  struct StringMaker<std::vector<T>>
  {
      static String convert(const std::vector<T>& vec)
      {
          std::ostringstream oss;
          oss << "[ ";
          std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(oss, " "));
          oss << "]";
          return oss.str().c_str();
      }
  };

  template <typename T>
  struct StringMaker<thrust::device_vector<T>>
  {
      static String convert(const thrust::device_vector<T>& vec)
      {
          std::ostringstream oss;
          oss << "[ ";
          std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(oss, " "));
          oss << "]";
          return oss.str().c_str();
      }
  };
}