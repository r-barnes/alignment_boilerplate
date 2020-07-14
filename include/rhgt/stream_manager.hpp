#include "ctpl.hpp"

#include <cuda_runtime.h>

#include <deque>
#include <functional>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace rhgt {

template<class GPUStorage, class TaskStorage>
class StreamManager;

typedef int gpu_id_t;

template<class GPUStorage, class TaskStorage>
struct StreamInfo {
  gpu_id_t     gpu_id;
  cudaStream_t stream_id;
  StreamManager<GPUStorage,TaskStorage> &stream_manager;
};



std::vector<std::pair<size_t, size_t>> generate_chunks(const size_t N, const size_t chunks){
  const int step = std::round(N/(double)chunks);

  std::vector<std::pair<size_t, size_t>> ret;

  for(size_t i=0;i<chunks;i++){
    ret.emplace_back(
      i*step,
      (i==chunks-1) ? N : (i+1)*step
    );
  }

  return ret;
}



std::vector<std::pair<size_t, size_t>> generate_chunks(const size_t begin, const size_t end, const size_t chunks){
  auto ret = generate_chunks(end-begin, chunks);

  for(auto &x: ret){
    x.first += begin;
    x.second += begin;
  }
  ret.back().second = end;

  return ret;
}


template<class GPUStorage, class TaskStorage>
class StreamManager {
 public:
  typedef std::function<GPUStorage(const size_t gbegin, const size_t gend)> GPUSetupFunc;
  typedef std::function<TaskStorage(const cudaStream_t stream, const size_t cbegin, const size_t cend)> TaskSetupFunc;
  typedef std::function<void(const cudaStream_t stream, TaskStorage &task_data)> TaskFinishFunc;

  StreamManager(const std::vector<gpu_id_t> &gpu_ids, const int streams_per_gpu, const int thread_pool_size);
  ~StreamManager();

  static void error_handle(const cudaError_t &err, const std::string &msg);

 private:
  const std::vector<gpu_id_t> gpu_ids;
  const int streams_per_gpu;

  typedef std::pair<TaskStorage,TaskFinishFunc> TaskSFPair;

  ctpl::thread_pool pool;
  std::vector<std::vector<cudaStream_t>> streams;
  std::vector<GPUStorage> gpu_data;
  std::vector<std::pair<size_t,size_t>> gpu_ranges;
  std::vector<std::unordered_map<cudaStream_t, std::deque<TaskSFPair>>> task_data;
  int _work_items;

  static void CUDART_CB cuda_stream_callback(cudaStream_t stream, cudaError_t status, void *userData);

  void setup_gpus(const int work_items, GPUSetupFunc setup_function);

  void run_tasks(TaskSetupFunc setup, TaskFinishFunc finish);

  cudaStream_t make_stream() const;
};


template<class GPUStorage, class TaskStorage>
StreamManager<GPUStorage,TaskStorage>::StreamManager(
  const std::vector<gpu_id_t> &gpu_ids,
  const int streams_per_gpu,
  const int thread_pool_size
) : gpu_ids(gpu_ids), streams_per_gpu(streams_per_gpu), pool(thread_pool_size) {
  streams.resize(gpu_ids.size());
  for(size_t gi=0;gi<gpu_ids.size();gi++)
  for(size_t si=0;si<streams_per_gpu;si++){
    streams.at(gi).push_back(make_stream());
  }

  gpu_data.resize  (gpu_ids.size());
  gpu_ranges.resize(gpu_ids.size());
  task_data.resize (gpu_ids.size());
}


template<class GPUStorage, class TaskStorage>
StreamManager<GPUStorage,TaskStorage>::~StreamManager(){
  for(const auto &gpu_streams: streams)
  for(const auto &stream: gpu_streams)
    error_handle(cudaStreamDestroy(stream), "Could not destroy a CUDA stream!");
}


template<class GPUStorage, class TaskStorage>
void StreamManager<GPUStorage,TaskStorage>::error_handle(const cudaError_t &err, const std::string &msg){
  if(err!=cudaSuccess)
    throw std::runtime_error(msg);
}


template<class GPUStorage, class TaskStorage>
cudaStream_t StreamManager<GPUStorage,TaskStorage>::make_stream() const {
  cudaStream_t temp;
  error_handle(cudaStreamCreate(&temp), "Could not create a CUDA stream!");
  error_handle(cudaStreamAddCallback(temp, cuda_stream_callback, (void*)this, 0), "Could not add a callback to CUDA stream!");
  return temp;
}


template<class GPUStorage, class TaskStorage>
void StreamManager<GPUStorage,TaskStorage>::setup_gpus(const int work_items, GPUSetupFunc setup_function){
  _work_items = work_items;

  const auto chunks = generate_chunks(work_items, gpu_ids.size());
  #pragma omp parallel for
  for(size_t i=0;i<gpu_ids.size();i++){
    error_handle(cudaSetDevice(gpu_ids.at(i)), "Failed to set CUDA device in setup_gpus!");
    gpu_data.at(i) = setup_function(chunks.at(i).first, chunks.at(i).second);
    gpu_ranges.at(i) = chunks.at(i);
  }
}


template<class GPUStorage, class TaskStorage>
void StreamManager<GPUStorage,TaskStorage>::run_tasks(TaskSetupFunc setup, TaskFinishFunc finish){
  #pragma omp parallel
  {
    const auto gpu_id    = gpu_ids.at(omp_get_thread_num());
    cudaSetDevice(gpu_id);

    const auto &gstreams = streams.at(omp_get_thread_num());
    const auto grange    = gpu_ranges.at(gpu_id);

    const auto chunks = generate_chunks(grange.first, grange.second, streams_per_gpu);

    size_t streami = 0;
    for(const auto &chunk: chunks){
      const auto &stream_id = gstreams.at(streami++%gstreams.size());
      task_data[stream_id].push_back({std::move(setup(
        stream_id,
        chunk.first,
        chunk.second
      )), finish});
      auto si = new StreamInfo<GPUStorage,TaskStorage>{gpu_id, stream_id, *this};
    }
  }
}

}