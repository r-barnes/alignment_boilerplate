#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>

namespace albp {

#define ALBP_CUDA_ERROR_CHECK(error) albp::cuda_error_handler(error, __FILE__, __LINE__)
#define ALBP_CHECK_KERNEL() albp::cuda_error_handler(cudaGetLastError(), __FILE__, __LINE__)

inline void cuda_error_handler(const cudaError_t &err, const std::string &file, const size_t line){
  if (err==cudaSuccess)
    return;

  const std::string msg = std::string("[albp CUDA ERROR:] ")
    + cudaGetErrorString(err)
    + " (CUDA error no.="+std::to_string(err)+")"
    + " in "+file+":"+std::to_string(line);

  throw std::runtime_error(msg);
}

}