#include <albp/doctest.hpp>
#include <albp/stream_manager.hpp>
#include <albp/memory.hpp>

using namespace albp;

namespace std {
  template<typename T, typename U>
  std::ostream& operator<< (std::ostream& os, const std::pair<T,U> &value) {
    os << "{" << value.first << "," << value.second << "}";
    return os;
  }
}


__global__ void gpu_test_kernel(double *const data, const int count){
  for (int di = blockIdx.x * blockDim.x + threadIdx.x; di < count; di += blockDim.x * gridDim.x){
    for(int i=0;i<20;i++)
      data[di] = sin(data[di]);
  }
}


TEST_CASE("Stream Manager"){
  const size_t work_items = 50'000;
  std::vector<double> data_vec;
  data_vec.reserve(work_items);

  for(size_t i=0;i<work_items;i++)
    data_vec.push_back(i%5);

  struct GPUStorage {
    double *data   = nullptr;
  };

  struct StreamStorage {
    double *d_data = nullptr;
  };

  const std::vector<int> gpu_ids = {0};
  const int streams_per_gpu  = 6;
  const int chunks_per_gpu   = 6;
  const int thread_pool_size = 6;

  GPUManager gpu_manager(gpu_ids, streams_per_gpu, thread_pool_size);

  const GPUSetupFunc<GPUStorage> gpu_setup = [&](const int gpu_id, const RangePair range, const size_t chunks, const size_t max_chunk_size) -> GPUStorage {
    (void)gpu_id; //Suppress unused variable warning
    GPUStorage gpu_storage;
    gpu_storage.data = PageLockedMalloc<double>(range.size());
    memcpy(gpu_storage.data, data_vec.data(), range.size()*sizeof(double));
    return gpu_storage;
  };

  const StreamSetupFunc<GPUStorage,StreamStorage> stream_setup = [&](const GPUStorage &gpu_storage, const cudaStream_t stream, const size_t max_chunk_size){
    StreamStorage temp;
    temp.d_data = DeviceMalloc<double>(max_chunk_size);
    return temp;
  };

  const TaskFunc<GPUStorage,StreamStorage> task_func = [](const GPUStorage &gpu_storage, const StreamStorage &stream_storage, const cudaStream_t stream, const RangePair range) {
    ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(stream_storage.d_data, gpu_storage.data+range.begin, range.size()*sizeof(double), cudaMemcpyHostToDevice, stream));

    gpu_test_kernel<<<200,128,0,stream>>>(stream_storage.d_data, range.size());
    ALBP_CUDA_ERROR_CHECK(cudaGetLastError());

    ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(gpu_storage.data+range.begin, stream_storage.d_data, range.size()*sizeof(double), cudaMemcpyDeviceToHost, stream));
  };

  const TaskFinishFunc<GPUStorage,StreamStorage> task_finish = [&](const GPUStorage &gpu_storage, const StreamStorage &stream_storage, const cudaStream_t stream, const RangePair range){};

  const StreamFinishFunc<GPUStorage,StreamStorage> stream_finish = [&](const GPUStorage &gpu_storage, StreamStorage &stream_storage, const cudaStream_t stream, const size_t max_chunk_size){
    ALBP_CUDA_ERROR_CHECK(cudaFree(stream_storage.d_data));
  };

  const GPUFinishFunc<GPUStorage> gpu_finish = [&](const int gpu_id, GPUStorage &gpu_storage){};

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

  ALBP_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  bool good = true;
  for(int wi=0;wi<work_items;wi++){
    double answer = wi%5;
    for(int i=0;i<20;i++)
      answer = std::sin(answer);
    good &= (answer==doctest::Approx(storage.at(0).gpu_storage.data[wi]));
  }

  CHECK(good);

  for(auto &kv: storage){
    ALBP_CUDA_ERROR_CHECK(cudaFreeHost(kv.second.gpu_storage.data));
  }
}
