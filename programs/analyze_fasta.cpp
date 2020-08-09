#include <albp/read_fasta.hpp>

#include <limits>
#include <iostream>

int main(int argc, char **argv){
  if(argc<2){
    std::cout<<argv[0]<<" <Fasta File> [Fasta File...]"<<std::endl;
    return -1;
  }

  std::cout<<"filename minlen maxlen avglen total_bytes\n";
  for(int i=1;i<argc;i++){
    const auto fasta = albp::ReadFasta(argv[i]);

    size_t minlen = std::numeric_limits<size_t>::max();
    size_t maxlen = 0;
    size_t total_len = 0;

    for(const auto &x: fasta.sequences){
      minlen = std::min(minlen, x.size());
      maxlen = std::max(maxlen, x.size());
      total_len += x.size();
    }

    std::cout<<argv[i]
             <<" "<<fasta.sequence_count()
             <<" "<<minlen
             <<" "<<maxlen
             <<" "<<(total_len/fasta.sequence_count())
             <<" "<<total_len
             <<"\n";
  }

  return 0;
}