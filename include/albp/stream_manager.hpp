#pragma once

#include <albp/functional.hpp>
#include <albp/ranges.hpp>

namespace albp {

cudaStream_t get_new_stream(const int device_id);

typedef fu2::function_view<void(const RangePair range)> StreamFunction;
typedef std::vector<StreamFunction> StreamVector;

void process_streams(StreamVector &sv, const RangePair range, const size_t chunk_size);
void process_streams(StreamVector &sv, const size_t item_count, const size_t chunk_size);

}
