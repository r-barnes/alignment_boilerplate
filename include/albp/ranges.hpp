#pragma once

#include <vector>

namespace albp {

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

}