#include <albp/doctest.hpp>
#include <albp/read_fasta.hpp>

#include <algorithm>
#include <stdexcept>

using namespace albp;

TEST_CASE("No fasta file"){
  CHECK_THROWS_AS(ReadFasta("not-a-file"), std::runtime_error);
}

//TODO: Test for modifiers reading correctly
TEST_CASE("Read Fasta"){
  FastaInput fasta;
  CHECK_NOTHROW(fasta=ReadFasta("target_batch.fasta"));

  CHECK(fasta.sequences.size()==20'000);
  CHECK(fasta.modifiers.size()==20'000);
  CHECK(fasta.headers.size()==20'000);
  CHECK(fasta.maximum_sequence_length==277);
  CHECK(fasta.sequence_count()==20'000);
  CHECK(fasta.sequences.at(5)=="TGGGATTAAAGATCCTGGACCGTGGCCAGGCGCGGCGGCTCAAGCCTGTAATCCCAGCGATCAGGGAGGCCGCCGCGGGAGGATTGCTTGAGCCCAGGAGTTTGAGACCAGCTTGGGCAACATAGCGAGACACCGTCTCTACAAAAAAATAACAAATAGTGGGGCGTGATGGCGCGCGCCTGTAGTCTCAGCTACTTGGGCGGTCGAGATGGGAGGATCGATCGAGTCTGGGAGGTCGAGGCTGCAGTGAGC");
}

TEST_CASE("Read Pair"){
  const auto input_data = ReadFastaQueryTargetPair("query_batch.fasta", "target_batch.fasta");
  CHECK(input_data.sequence_count()==20'000);
}

TEST_CASE("get_max_length"){
  std::vector<std::string> dat = {
    "abc",
    "abcdef",
    "g",
    "asdfa"
  };

  std::sort(dat.begin(), dat.end());

  do {
    CHECK(get_max_length(dat)==6);
  } while (next_permutation(dat.begin(), dat.end()));
}