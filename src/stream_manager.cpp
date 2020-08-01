#include <albp/error_handling.hpp>
#include <albp/stream_manager.hpp>

namespace albp {

GPUManager::GPUManager(
  const std::vector<gpu_id_t> &gpu_ids,
  const size_t streams_per_gpu,
  const size_t thread_pool_size
) : pool(thread_pool_size) {
  for(const auto &gpu_id: gpu_ids)
  for(size_t si=0;si<streams_per_gpu;si++){
    gpu_streams[gpu_id].push_back(make_stream(gpu_id));
  }
}


GPUManager::~GPUManager(){
  for(const auto &gpu_stream: gpu_streams)
  for(const auto &stream: gpu_stream.second)
    RCHECKCUDAERROR(cudaStreamDestroy(stream));
}


cudaStream_t GPUManager::make_stream(const gpu_id_t gpu_id) const {
  cudaStream_t temp;
  RCHECKCUDAERROR(cudaSetDevice(gpu_id));
  RCHECKCUDAERROR(cudaStreamCreate(&temp));
  return temp;
}

}