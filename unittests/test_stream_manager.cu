#include <rhgt/stream_manager.hpp>
#include <rhgt/memory.hpp>

#include "doctest.h"

using namespace rhgt;

namespace std {
  template<typename T, typename U>
  std::ostream& operator<< (std::ostream& os, const std::pair<T,U> &value) {
    os << "{" << value.first << "," << value.second << "}";
    return os;
  }
}

TEST_CASE("Generate Chunks"){
  SUBCASE("One Chunk"){
    const auto chunks = generate_chunks(20, 1);
    CHECK(chunks.size()==1);
    CHECK(chunks.at(0)==std::make_pair<size_t,size_t>(0, 20));
  }

  SUBCASE("Even"){
    const auto chunks = generate_chunks(20, 5);
    CHECK(chunks.size()==5);
    CHECK(chunks.at(0)==std::make_pair<size_t,size_t>( 0, 4));
    CHECK(chunks.at(1)==std::make_pair<size_t,size_t>( 4, 8));
    CHECK(chunks.at(2)==std::make_pair<size_t,size_t>( 8,12));
    CHECK(chunks.at(3)==std::make_pair<size_t,size_t>(12,16));
    CHECK(chunks.at(4)==std::make_pair<size_t,size_t>(16,20));
  }

  SUBCASE("Too long for last"){
    const auto chunks = generate_chunks(21, 5);
    CHECK(chunks.size()==5);
    CHECK(chunks.at(0)==std::make_pair<size_t,size_t>( 0, 4));
    CHECK(chunks.at(1)==std::make_pair<size_t,size_t>( 4, 8));
    CHECK(chunks.at(2)==std::make_pair<size_t,size_t>( 8,12));
    CHECK(chunks.at(3)==std::make_pair<size_t,size_t>(12,16));
    CHECK(chunks.at(4)==std::make_pair<size_t,size_t>(16,21));
  }

  SUBCASE("Too short for last"){
    const auto chunks = generate_chunks(19, 5);
    CHECK(chunks.size()==5);
    CHECK(chunks.at(0)==std::make_pair<size_t,size_t>( 0, 4));
    CHECK(chunks.at(1)==std::make_pair<size_t,size_t>( 4, 8));
    CHECK(chunks.at(2)==std::make_pair<size_t,size_t>( 8,12));
    CHECK(chunks.at(3)==std::make_pair<size_t,size_t>(12,16));
    CHECK(chunks.at(4)==std::make_pair<size_t,size_t>(16,19));
  }

  SUBCASE("Too short for last and below 0.5 point"){
    const auto chunks = generate_chunks(17, 5);
    CHECK(chunks.size()==5);
    CHECK(chunks.at(0)==std::make_pair<size_t,size_t>( 0, 3));
    CHECK(chunks.at(1)==std::make_pair<size_t,size_t>( 3, 6));
    CHECK(chunks.at(2)==std::make_pair<size_t,size_t>( 6, 9));
    CHECK(chunks.at(3)==std::make_pair<size_t,size_t>( 9,12));
    CHECK(chunks.at(4)==std::make_pair<size_t,size_t>(12,17));
  }
}

TEST_CASE("Thread Pool"){
  ctpl::thread_pool pool(4);
  auto ret1 = pool.push([](const int thread_id) -> int { return 1; });
  auto ret2 = pool.push([](const int thread_id) -> int { return 2; });
  auto ret3 = pool.push([](const int thread_id) -> int { return 3; });
  auto ret4 = pool.push([](const int thread_id) -> int { return 4; });
  CHECK(ret1.get()==1);
}



TEST_CASE("Stream Manager"){
  const int N = 30000;
  std::vector<int> data;
  data.reserve(N);

  for(int i=0;i<N;i++)
    data.push_back(i);

  class GPUStorage {
    int *data;
  };

  class TaskStorage {};

  const std::vector<int> gpu_ids = {0};
  const int streams_per_gpu = 6;
  const int thread_pool_size = 4;

  typedef StreamManager<GPUStorage,TaskStorage> SM;

  SM sm(gpu_ids, streams_per_gpu, thread_pool_size);

  const SM::GPUSetupFunc gpu_setup = [&](const size_t start, const size_t end) -> GPUStorage {
    GPUStorage temp;
    const int count = end-start;
    temp.data = PageLockedMalloc<int>(count);
    return temp;
  };


  sm.setup_gpus(N, gpu_setup);
}
