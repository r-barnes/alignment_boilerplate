#pragma once

#include <albp/error_handling.hpp>
#include <albp/ranges.hpp>

#include <iostream>
#include <memory>
#include <string>

namespace albp {

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

//Allocate `count` items in page-locked memory
template<class T>
T* PageLockedMalloc(const size_t count, const T *const host_data=nullptr){
    T *ptr;
    ALBP_CUDA_ERROR_CHECK(cudaMallocHost(&ptr, count*sizeof(T), cudaHostAllocDefault));
    if(host_data)
        ALBP_CUDA_ERROR_CHECK(cudaMemcpy(ptr, host_data, count*sizeof(T), cudaMemcpyHostToHost));
    return ptr;
}

//Allocate `count` items in page-locked memory
template<class T>
cuda_unique_hptr<T> PageLockedMallocUnique(const size_t count, const T *const host_data=nullptr){
    T *ptr;
    ALBP_CUDA_ERROR_CHECK(cudaMallocHost(&ptr, count*sizeof(T), cudaHostAllocDefault));
    if(host_data)
        ALBP_CUDA_ERROR_CHECK(cudaMemcpy(ptr, host_data, count*sizeof(T), cudaMemcpyHostToHost));
    return cuda_unique_hptr<T>(ptr);
}

//Allocate `count` items on device memory
template<class T>
T* DeviceMalloc(const size_t count, const T *const host_data=nullptr){
    T *ptr;
    ALBP_CUDA_ERROR_CHECK(cudaMalloc(&ptr, count*sizeof(T)));
    if(host_data)
        ALBP_CUDA_ERROR_CHECK(cudaMemcpy(ptr, host_data, count*sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

//Allocate `count` items on device memory
template<class T>
cuda_unique_dptr<T> DeviceMallocUnique(const size_t count, const T *const host_data=nullptr){
    T *ptr;
    ALBP_CUDA_ERROR_CHECK(cudaMalloc(&ptr, count*sizeof(T)));
    if(host_data)
        ALBP_CUDA_ERROR_CHECK(cudaMemcpy(ptr, host_data, count*sizeof(T), cudaMemcpyHostToDevice));
    return cuda_unique_dptr<T>(ptr);
}

///Copies `count` items from `host_ptr` to `dev_ptr` asynchronously
template<class T>
void copy_to_device_async(T *dev_ptr, const T *host_ptr, const size_t count, const cudaStream_t stream){
  ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(dev_ptr, host_ptr, count*sizeof(T), cudaMemcpyHostToDevice, stream));
}

///Copies `count` items from `dev_ptr` to `host_ptr` asynchronously
template<class T>
void copy_to_host_async(T *host_ptr, const T *dev_ptr, const size_t count, const cudaStream_t stream){
  ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(host_ptr, dev_ptr, count*sizeof(T), cudaMemcpyDeviceToHost, stream));
}

///Copies `count` items from `host_ptr` to `dev_ptr` asynchronously
template<class T>
void copy_to_device_async(T *dev_ptr, const T *host_ptr, const RangePair &rp, const cudaStream_t stream){
  ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(dev_ptr, &host_ptr[rp.begin], rp.size()*sizeof(T), cudaMemcpyHostToDevice, stream));
}

///Copies `count` items from `dev_ptr` to `host_ptr` asynchronously
template<class T>
void copy_to_host_async(T *host_ptr, const T *dev_ptr, const RangePair &rp, const cudaStream_t stream){
  ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_ptr[rp.begin], dev_ptr, rp.size()*sizeof(T), cudaMemcpyDeviceToHost, stream));
}

}