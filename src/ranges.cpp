#include <cmath>
#include <albp/ranges.hpp>

namespace albp {

RangeVector generate_chunks(const size_t N, const size_t chunks){
  const int step = std::round(N/(double)chunks);

  RangeVector ret;

  for(size_t i=0;i<chunks;i++){
    ret.emplace_back(
      i*step,
      (i==chunks-1) ? N : (i+1)*step
    );
  }

  return ret;
}


RangeVector generate_chunks(const size_t begin, const size_t end, const size_t chunks){
  auto ret = generate_chunks(end-begin, chunks);

  for(auto &x: ret){
    x.begin += begin;
    x.end   += begin;
  }
  ret.back().end = end;

  return ret;
}


RangeVector generate_chunks(const RangePair &range_pair, const size_t chunks){
  return generate_chunks(range_pair.begin, range_pair.end, chunks);
}


size_t get_maximum_range(const RangeVector &ranges){
  if(ranges.empty())
    return 0;
  //The largest range is either the first one or the last one because all the
  //ranges except the last one are the same size as the first one.
  return std::max(ranges.front().size(), ranges.back().size());
}

}