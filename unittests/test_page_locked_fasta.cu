#include <albp/doctest.hpp>
#include <albp/page_locked_fasta.hpp>
#include <albp/read_fasta.hpp>

using namespace albp;

TEST_CASE("page_lock"){
  FastaInput fasta;
  CHECK_NOTHROW(fasta=ReadFasta("target_batch.fasta"));

  const auto pl_fasta = page_lock(fasta);

  CHECK(pl_fasta.sequence_count==20'000);
  CHECK(pl_fasta.maximum_sequence_length==277);
  CHECK(pl_fasta.headers==fasta.headers);
  CHECK(pl_fasta.sequences.size()==fasta.total_sequence_bytes);
  CHECK(pl_fasta.sequences.capacity()==fasta.total_sequence_bytes);

  for(size_t i=0;i<pl_fasta.sequence_count;i++){
    CHECK(pl_fasta.sizes[i]==fasta.sequences.at(i).size());
  }

  for(size_t i=0;i<pl_fasta.sequence_count;i++){
    CHECK(0==fasta.sequences.at(i).compare(
      0,
      std::string::npos,
      &pl_fasta.sequences[pl_fasta.starts[i]],
      pl_fasta.sizes[i])
    );

    CHECK(0==fasta.sequences.at(i).compare(
      0,
      std::string::npos,
      &pl_fasta.sequences[pl_fasta.starts[i]],
      pl_fasta.ends[i]-pl_fasta.starts[i])
    );
  }
}