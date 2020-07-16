#include <rhgb/error_handling.hpp>
#include <rhgb/stream_manager.hpp>

namespace rhgb {

RangeVector generate_chunks(const size_t N, const size_t chunks){
  const int step = std::round(N/(double)chunks);

  RangeVector ret;

  for(size_t i=0;i<chunks;i++){
    ret.emplace_back(
      i*step,
      (i==chunks-1) ? N : (i+1)*step
    );
  }

  return ret;
}


RangeVector generate_chunks(const size_t begin, const size_t end, const size_t chunks){
  auto ret = generate_chunks(end-begin, chunks);

  for(auto &x: ret){
    x.begin += begin;
    x.end   += begin;
  }
  ret.back().end = end;

  return ret;
}


RangeVector generate_chunks(const RangePair &range_pair, const size_t chunks){
  return generate_chunks(range_pair.begin, range_pair.end, chunks);
}


size_t get_maximum_range(const RangeVector &ranges){
  if(ranges.empty())
    return 0;
  //The largest range is either the first one or the last one because all the
  //ranges except the last one are the same size as the first one.
  return std::max(ranges.front().size(), ranges.back().size());
}





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