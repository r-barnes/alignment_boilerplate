#pragma once

#include "error_handling.hpp"

#include <iostream>
#include <memory>
#include <string>

namespace albp {

//Allocate `count` items in page-locked memory
template<class T>
T* PageLockedMalloc(const size_t count, const T *const host_data=nullptr){
    T *temp;
    ALBP_CUDA_ERROR_CHECK(cudaMallocHost(&temp, count*sizeof(T), cudaHostAllocDefault));
    if(host_data)
        ALBP_CUDA_ERROR_CHECK(cudaMemcpy(temp, host_data, count*sizeof(T), cudaMemcpyHostToHost));
    return temp;
}

//Allocate `count` items on device memory
template<class T>
T* DeviceMalloc(const size_t count, const T *const host_data=nullptr){
    T *temp;
    ALBP_CUDA_ERROR_CHECK(cudaMalloc(&temp, count*sizeof(T)));
    if(host_data)
        ALBP_CUDA_ERROR_CHECK(cudaMemcpy(temp, host_data, count*sizeof(T), cudaMemcpyHostToDevice));
    return temp;
}

template<typename T>
struct cuda_device_deleter {
  inline void operator()( T* ptr ) const {
    if(ptr)
      cudaFree(ptr);
  }
};

template<typename T>
struct cuda_host_deleter {
  inline void operator()( T* ptr ) const {
    if(ptr)
      cudaFreeHost(ptr);
  }
};

template<class T>
using cuda_unique_hptr = std::unique_ptr<T[], cuda_host_deleter<T>>;

template<class T>
using cuda_unique_dptr = std::unique_ptr<T[], cuda_device_deleter<T>>;

}