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
  //Break up the input range into chunks
  const auto chunks = generate_chunks_of_size(range, chunk_size);

  //Divide the chunks among the streams
  const auto chunks_to_streams = generate_n_chunks(chunks.size(), sv.size());

  std::cerr<<"chunks = "<<chunks.size()<<std::endl;

  std::vector<std::thread> threads(sv.size());

  //Loop through all streams
  for(size_t si=0;si<sv.size();si++){
    threads.at(si) = std::thread([&, si](){
      for(size_t ci=chunks_to_streams.at(si).begin; ci<chunks_to_streams.at(si).end;ci++){
        sv.at(si)(chunks.at(ci));
      }
    });
  }

  //Wait for CUDA to finish
  cudaDeviceSynchronize();

  //Wait for all threads to finish
  for(auto &thread: threads){
    thread.join();
  }
}

}