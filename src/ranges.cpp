#include <albp/ranges.hpp>

#include <cmath>
#include <stdexcept>

namespace albp {

RangeVector generate_n_chunks(const size_t N, const size_t chunk_count){
  const int step = std::round(N/(double)chunk_count);

  RangeVector ret;

  for(size_t i=0;i<chunk_count;i++){
    ret.emplace_back(
      i*step,
      (i==chunk_count-1) ? N : (i+1)*step
    );
  }

  return ret;
}


RangeVector generate_n_chunks(const size_t begin, const size_t end, const size_t chunk_count){
  auto ret = generate_n_chunks(end-begin, chunk_count);

  for(auto &x: ret){
    x.begin += begin;
    x.end   += begin;
  }
  ret.back().end = end;

  return ret;
}


RangeVector generate_n_chunks(const RangePair &range_pair, const size_t chunk_count){
  return generate_n_chunks(range_pair.begin, range_pair.end, chunk_count);
}

RangeVector generate_chunks_of_size(const size_t N, const size_t chunk_size){
  return generate_chunks_of_size(RangePair(0,N), chunk_size);
}

RangeVector generate_chunks_of_size(const RangePair &range_pair, const size_t chunk_size){
  RangeVector ret;

  if(chunk_size==0){
    throw std::runtime_error("chunk_size must be >0!");
  }

  for(size_t i=range_pair.begin;i<range_pair.end;i+=chunk_size){
    ret.emplace_back(
      i,
      std::min(range_pair.end, i+chunk_size)
    );
  }

  return ret;
}


size_t get_maximum_range(const RangeVector &ranges){
  if(ranges.empty())
    return 0;
  //The largest range is either the first one or the last one because all the
  //ranges except the last one are the same size as the first one.
  return std::max(ranges.front().size(), ranges.back().size());
}

}