#include <albp/page_locked_fasta.hpp>
#include <albp/memory.hpp>

namespace albp {

PageLockedFasta::PageLockedFasta(const size_t sequence_count, const size_t sequence_bytes) :
  sequence_count(sequence_count), sequences(sequence_bytes), modifiers(sequence_count)
{
  starts.reset(PageLockedMalloc<size_t>(sequence_count));
  ends.reset  (PageLockedMalloc<size_t>(sequence_count));
  sizes.reset (PageLockedMalloc<size_t>(sequence_count));
}

char *       PageLockedFasta::seq_begin(size_t i)       { return &sequences[starts[i]]; }
const char * PageLockedFasta::seq_begin(size_t i) const { return &sequences[starts[i]]; }
char *       PageLockedFasta::seq_end  (size_t i)       { return &sequences[ends[i]];   }
const char * PageLockedFasta::seq_end  (size_t i) const { return &sequences[ends[i]];   }



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

}