#pragma once

#include <vector>

namespace albp {

///A half-open range [begin,end)
struct RangePair {
  size_t begin; //< Beginning of the range (inclusive)
  size_t end;   //< End of the range (exclusive)
  RangePair() = default;
  RangePair(size_t begin, size_t end) : begin(begin), end(end) {}
  ///Number of elements in the range [begin,end)
  size_t size() const {
    return end-begin;
  }
  ///Determine whether two ranges are equal to each other
  bool operator==(const RangePair &o) const {
    return begin==o.begin && end==o.end;
  }
};

typedef std::vector<RangePair> RangeVector;

RangeVector generate_n_chunks(const size_t N, const size_t chunk_count);
RangeVector generate_n_chunks(const size_t begin, const size_t end, const size_t chunk_count);
RangeVector generate_n_chunks(const RangePair &range_pair, const size_t chunk_count);
RangeVector generate_chunks_of_size(const size_t N, const size_t chunk_size);
RangeVector generate_chunks_of_size(const RangePair &range_pair, const size_t chunk_size);

size_t get_maximum_range(const RangeVector &ranges);

}