#include "ctpl.hpp"
#include "error_handling.hpp"

#include <cuda_runtime.h>

#include <iostream>

#include <cmath>
#include <deque>
#include <functional>
#include <omp.h>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace rhgt {

typedef int gpu_id_t;

struct RangePair {
  size_t begin;
  size_t end;
  RangePair() = default;
  RangePair(size_t begin, size_t end) : begin(begin), end(end) {}
  size_t size() const {
    return end-begin;
  }
  bool operator==(const RangePair &o) const {
    return begin==o.begin && end==o.end;
  }
};

typedef std::vector<RangePair> RangeVector;

RangeVector generate_chunks(const size_t N, const size_t chunks);
RangeVector generate_chunks(const size_t begin, const size_t end, const size_t chunks);
RangeVector generate_chunks(const RangePair &range_pair, const size_t chunks);

size_t get_maximum_range(const RangeVector &ranges);


class GPUManager {
 public:
  GPUManager(const std::vector<gpu_id_t> &gpu_ids, const size_t streams_per_gpu, const size_t thread_pool_size);
  ~GPUManager();
  std::unordered_map<gpu_id_t, std::vector<cudaStream_t>> gpu_streams;
  ctpl::thread_pool pool;

 private:
  cudaStream_t make_stream(const gpu_id_t gpu_id) const;
};



template<typename GPUStorage>
using GPUSetupFunc = std::function<GPUStorage(const int gpu_id, const RangePair range, const size_t chunks, const size_t max_chunk_size)>;

template<typename GPUStorage, typename StreamStorage>
using StreamSetupFunc = std::function<StreamStorage(const GPUStorage &gpu_storage, const cudaStream_t stream, const size_t max_chunk_size)>;

template<typename GPUStorage, typename StreamStorage>
using TaskFunc = std::function<void(const GPUStorage &gpu_storage, const StreamStorage &stream_storage, const cudaStream_t stream, const RangePair range)>;

template<typename GPUStorage, typename StreamStorage>
using TaskFinishFunc = std::function<void(const GPUStorage &gpu_storage, const StreamStorage &stream_storage, const cudaStream_t stream, const RangePair range)>;

template<typename GPUStorage, typename StreamStorage>
using StreamFinishFunc = std::function<void(const GPUStorage &gpu_storage, StreamStorage &stream_storage, const cudaStream_t stream, const size_t max_chunk_size)>;

template<typename GPUStorage>
using GPUFinishFunc = std::function<void(const int gpu_id, GPUStorage &gpu_storage)>;



template<typename GPUStorage, typename StreamStorage>
struct PerDeviceStorage {
  GPUStorage gpu_storage;
  std::unordered_map<cudaStream_t, StreamStorage> streams;
};

template<typename GPUStorage, typename StreamStorage>
using DeviceStorage = std::unordered_map<gpu_id_t, PerDeviceStorage<GPUStorage,StreamStorage>>;




/* Call with

  const GPUSetupFunc<GPUStorage> gpu_setup = [&](const int gpu_id, const RangePair range, const size_t chunks, const size_t max_chunk_size) -> GPUStorage {

  };

  const StreamSetupFunc<GPUStorage,StreamStorage> stream_setup = [&](const GPUStorage &gpu_storage, const cudaStream_t stream, const size_t max_chunk_size){

  };

  const TaskFunc<GPUStorage,StreamStorage> task_func = [](const GPUStorage &gpu_storage, const StreamStorage &stream_storage, const cudaStream_t stream, const RangePair range) {

  };

  const TaskFinishFunc<GPUStorage,StreamStorage> task_finish = [&](const GPUStorage &gpu_storage, const StreamStorage &stream_storage, const cudaStream_t stream, const RangePair range){

  };

  const StreamFinishFunc<GPUStorage,StreamStorage> stream_finish = [&](const GPUStorage &gpu_storage, StreamStorage &stream_storage, const cudaStream_t stream, const size_t max_chunk_size){

  };

  const GPUFinishFunc<GPUStorage> gpu_finish = [&](const int gpu_id, GPUStorage &gpu_storage){

  };

  const auto storage = TaskWork(
    gpu_manager,
    work_items,
    chunks_per_gpu,
    gpu_setup,
    stream_setup,
    task_func,
    task_finish,
    stream_finish,
    gpu_finish
  );
*/
template<typename GPUStorage, typename StreamStorage>
DeviceStorage<GPUStorage,StreamStorage> TaskWork(
  GPUManager  &gpu_manager,
  const int    work_items,
  const int    chunks_per_gpu,
  GPUSetupFunc<GPUStorage>                    gpu_setup,
  StreamSetupFunc<GPUStorage,StreamStorage>   stream_setup,
  TaskFunc<GPUStorage,StreamStorage>          task_func,
  TaskFinishFunc<GPUStorage,StreamStorage>    task_finish,
  StreamFinishFunc<GPUStorage,StreamStorage>  stream_finish,
  GPUFinishFunc<GPUStorage>                   gpu_finish
){
  const auto gpu_chunks = generate_chunks(work_items, gpu_manager.gpu_streams.size());

  //Allocate per-device and per-stream storage
  DeviceStorage<GPUStorage,StreamStorage> storage;

  //Set up the unordered_map for all the GPUs so we can access it with mutex in the threads below
  for(const auto &gpu_stream: gpu_manager.gpu_streams){
    storage[gpu_stream.first] = {};
    for(const auto &stream: gpu_stream.second)
      storage.at(gpu_stream.first).streams[stream] = {};
  }

  //Vector for handling multithreading of GPU management
  std::vector<std::thread> gpu_runners;

  //Vector for handling each stream on each GPU
  std::vector<std::vector<std::thread>> gpu_stream_runners(gpu_manager.gpu_streams.size());

  //Consider each GPU
  size_t gi = 0; //GPU index - used for mapping GPUs to chunks
  for(auto &gpu_stream: gpu_manager.gpu_streams){
    const auto gpu_id = gpu_stream.first;       //CUDA ID of this GPU

    //Set the GPUs up in separate threads
    gpu_runners.emplace_back([&, gi, gpu_id](){
      const auto &stream_ids = gpu_manager.gpu_streams.at(gpu_id); //Streams assigned to this GPU
      auto &stream_runners = gpu_stream_runners.at(gi);
      auto &gpu_storage = storage.at(gpu_id).gpu_storage;

      //Switch this thread to this GPU
      RCHECKCUDAERROR(cudaSetDevice(gpu_id));

      //Break this GPU's chunk up into chunks to be assigned to streams
      const auto this_gpu_chunks = generate_chunks(gpu_chunks.at(gi), chunks_per_gpu);

      //Determine the maximum size of any chunk assigned to this GPU
      const auto max_chunk_size = get_maximum_range(this_gpu_chunks);

      //Determine which chunks belong to which streams
      const auto chunks_to_streams = generate_chunks(this_gpu_chunks.size(), stream_ids.size());

      //Setup memory for each GPU
      if(gpu_setup)
        gpu_storage = gpu_setup(gpu_id, gpu_chunks.at(gi), chunks_per_gpu, max_chunk_size);

      //Handle all the streams
      size_t stream_idx = 0;
      for(const auto &stream_id: stream_ids){
        stream_runners.emplace_back([&, stream_idx](){
          auto &stream_storage = storage.at(gpu_id).streams.at(stream_id);
          if(stream_setup)
            stream_storage = stream_setup(gpu_storage, stream_id, max_chunk_size);

          const auto my_chunks = chunks_to_streams.at(stream_idx);
          for(size_t mci=my_chunks.begin;mci<my_chunks.end;mci++){
            const auto this_chunk = this_gpu_chunks.at(mci);
            if(task_func)
              task_func(gpu_storage, stream_storage, stream_id, this_chunk);
            if(task_finish){
              cudaStreamSynchronize(stream_id);
              task_finish(gpu_storage, stream_storage, stream_id, this_chunk);
            }
          }
          cudaStreamSynchronize(stream_id);
          if(stream_finish)
            stream_finish(gpu_storage, stream_storage, stream_id, max_chunk_size);
        });
        stream_idx++;
      }

      //Wait for all streams to finish running
      for(auto &x: stream_runners)
        x.join();
    });
    gi++;
  }

  //Wait for processing to finish
  for(auto &x: gpu_runners)
    x.join();
  gpu_runners.clear();

  //Wait for all the GPUs to finish
  for(const auto &kv: gpu_manager.gpu_streams){
    RCHECKCUDAERROR(cudaSetDevice(kv.first));
    RCHECKCUDAERROR(cudaDeviceSynchronize());
  }

  //Clean up afterwards
  if(gpu_finish){
    for(auto &gpu_stream: gpu_manager.gpu_streams){
      const auto gpu_id = gpu_stream.first;
      gpu_runners.emplace_back([&,gpu_id](){
        RCHECKCUDAERROR(cudaSetDevice(gpu_id));
        if(gpu_finish)
          gpu_finish(gpu_id, storage.at(gpu_id).gpu_storage);
      });
    }
  }

  //Wait for cleanup to complete
  for(auto &x: gpu_runners)
    x.join();
  gpu_runners.clear();

  //Return the storage, in case the user wants that
  return storage;
}

}