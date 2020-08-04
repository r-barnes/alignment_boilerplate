#include <albp/page_locked_fasta.hpp>
#include <albp/memory.hpp>

namespace albp {

PageLockedFasta::PageLockedFasta(const size_t sequence_count, const size_t sequence_bytes) :
  sequence_count(sequence_count), sequences(sequence_bytes), modifiers(sequence_count)
{
  starts.reset(PageLockedMalloc<size_t>(sequence_count+1));
  ends.reset  (PageLockedMalloc<size_t>(sequence_count  ));
  sizes.reset (PageLockedMalloc<size_t>(sequence_count  ));
}

char *       PageLockedFasta::seq_begin(size_t i)       { return &sequences[starts[i]]; }
const char * PageLockedFasta::seq_begin(size_t i) const { return &sequences[starts[i]]; }
char *       PageLockedFasta::seq_end  (size_t i)       { return &sequences[ends[i]];   }
const char * PageLockedFasta::seq_end  (size_t i) const { return &sequences[ends[i]];   }

size_t PageLockedFasta::bytes_between(size_t a, size_t b) const {
  return ends[b-1]-starts[a];
}

size_t PageLockedFasta::total_bytes() const {
  return ends[sequence_count-1] - starts[0];
}

size_t PageLockedFasta::bytes_between(const RangePair &rp) const {
  return bytes_between(rp.begin, rp.end);
}



PageLockedFasta page_lock(const FastaInput &inp){
  PageLockedFasta ret(inp.sequence_count(), inp.total_sequence_bytes);

  ret.maximum_sequence_length = inp.maximum_sequence_length;

  //Move all sequences into page-locked memory
  for(size_t i=0;i<inp.sequence_count();i++){
    ret.sequences += inp.sequences.at(i);
    if(i==0){
      ret.starts[i] = 0;
      ret.ends[i]   = inp.sequences.at(i).size();
    } else {
      ret.starts[i] = ret.ends[i-1];
      ret.ends[i]   = ret.starts[i] + inp.sequences.at(i).size();
    }
    ret.sizes[i]  = inp.sequences.at(i).size();
  }
  ret.starts[inp.sequence_count()] = ret.ends[inp.sequence_count()-1];

  ret.headers = inp.headers;

  for(const auto &x: inp.modifiers)
    ret.modifiers += x;

  return ret;
}



PageLockedFastaPair page_lock(const FastaPair &fp){
  return PageLockedFastaPair{
    page_lock(fp.a),
    page_lock(fp.b),
    fp.total_cells_1_to_1(),
    fp.sequence_count()
  };
}



size_t get_max_length(const PageLockedFasta &pl_fasta, const RangePair range){
  size_t max_len = 0;
  for(size_t i=range.begin;i<range.end;i++){
    max_len = std::max(max_len, pl_fasta.sizes[i]);
  }
  return max_len;
}



size_t get_max_length(const PageLockedFasta &pl_fasta){
  return get_max_length(pl_fasta, RangePair(0, pl_fasta.sequence_count));
}



void copy_sequences_to_device_async(char *dev_ptr, const PageLockedFasta &pl_fasta, const RangePair &rp, const cudaStream_t stream){
  ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(dev_ptr, pl_fasta.seq_begin(rp.begin), pl_fasta.bytes_between(rp)*sizeof(char), cudaMemcpyHostToDevice, stream));
}

void copy_sequences_to_device_async(const cuda_unique_dptr<char> &dev_ptr, const PageLockedFasta &pl_fasta, const RangePair &rp, const cudaStream_t stream){
  copy_sequences_to_device_async(dev_ptr.get(), pl_fasta, rp, stream);
}

}