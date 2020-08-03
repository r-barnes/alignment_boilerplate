#include <albp/ctpl.hpp>
#include <albp/error_handling.hpp>
#include <albp/ranges.hpp>
#include <albp/stream_manager.hpp>

#include <cuda_runtime.h>

#include <thread>
#include <chrono>
#include <iostream>

namespace albp {

cudaStream_t get_new_stream(const int device_id){
  ALBP_CUDA_ERROR_CHECK(cudaSetDevice(device_id));
  cudaStream_t temp;
  ALBP_CUDA_ERROR_CHECK(cudaStreamCreate(&temp));
  return temp;
}



void process_streams(StreamVector &sv, const size_t item_count, const size_t chunk_size){
  process_streams(sv, RangePair(0, item_count), chunk_size);
}

void process_streams(StreamVector &sv, const RangePair range, const size_t chunk_size){
  //One thread per stream
  ctpl::thread_pool pool(sv.size());

  //Break up the input range into chunks
  const auto chunks = generate_chunks(range, 1);

  std::cerr<<"chunks = "<<chunks.size()<<std::endl;

  //Assign chunks to the streams round-robin style
  for(size_t i=0;i<chunks.size();i++){
    pool.push([&](const int){
      sv.at(i%sv.size())(chunks.at(i));
    });
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));

  //Wait for all the tasks to finish and then close out the threads
  pool.stop(true);
}

}