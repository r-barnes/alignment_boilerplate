#pragma once

#include <albp/ranges.hpp>

#include <string>
#include <vector>

namespace albp {

struct FastaInput {
  std::vector<std::string> sequences; //< Sequences in the file
  std::vector<std::string> headers;   //< Names/sequence identifiers
  std::vector<uint8_t>     modifiers; //< Character starting the new sequence, usually '>'
  size_t maximum_sequence_length = 0; //< Length of the longest sequence in the file
  size_t total_sequence_bytes    = 0; //< Number of bytes in all the sequences combined
  size_t sequence_count() const;      //< Number of sequences
};

struct FastaPair {
  FastaInput a;                        //< Sequences from the first file
  FastaInput b;                        //< Sequences from the second file
  ///@brief Calculates how many cells need to be calculated if the sequences are compared in pairs
  uint64_t total_cells_1_to_1() const;
  ///@brief Returns the number of sequences (each file has the same number)
  size_t sequence_count() const;
};

///@brief Reads FASTA files and makes a note of their sequences, headers, and modifer characters;
///
///@param[in] filename Filename of the FASTA file to read
///
///FASTA files contain sequences that are usually on separate lines. The file
///reader detects a '>' then concatenates all the following lines into one
///sequence, until the next '>' or EOF.
///
///See more about FASTA format : https://en.wikipedia.org/wiki/FASTA_format
FastaInput ReadFasta(const std::string &filename);

FastaPair ReadFastaQueryTargetPair(const std::string &query, const std::string &target);

///@brief Given a vector of strings, return the length of the longest string
///
///@param[in] vector_of_strings A vector of strings
///
///@returns The length of the longest string
size_t get_max_length(const std::vector<std::string> &vector_of_strings);

///@brief Given a vector of strings, return the length of the longest string
///
///@param[in] vector_of_strings A vector of strings
///@param[in] range Range of strings to consider
///
///@returns The length of the longest string
size_t get_max_length(const std::vector<std::string> &vector_of_strings, const RangePair range);

}