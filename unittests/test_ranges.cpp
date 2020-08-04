#include "doctest.h"

#include <albp/ranges.hpp>

#include <stdexcept>

using namespace albp;

TEST_CASE("generate_n_chunks"){
  SUBCASE("Zero Chunk"){
    const auto chunks = generate_n_chunks(20, 1);
    CHECK(chunks.size()==1);
    CHECK(get_maximum_range(chunks)==20);
    CHECK(chunks.at(0)==RangePair(0, 20));
  }

  SUBCASE("One Chunk"){
    const auto chunks = generate_n_chunks(20, 1);
    CHECK(chunks.size()==1);
    CHECK(get_maximum_range(chunks)==20);
    CHECK(chunks.at(0)==RangePair(0, 20));
  }

  SUBCASE("Even"){
    const auto chunks = generate_n_chunks(20, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==4);
    CHECK(chunks.at(0)==RangePair( 0, 4));
    CHECK(chunks.at(1)==RangePair( 4, 8));
    CHECK(chunks.at(2)==RangePair( 8,12));
    CHECK(chunks.at(3)==RangePair(12,16));
    CHECK(chunks.at(4)==RangePair(16,20));
  }

  SUBCASE("Too long for last"){
    const auto chunks = generate_n_chunks(21, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==5);
    CHECK(chunks.at(0)==RangePair( 0, 4));
    CHECK(chunks.at(1)==RangePair( 4, 8));
    CHECK(chunks.at(2)==RangePair( 8,12));
    CHECK(chunks.at(3)==RangePair(12,16));
    CHECK(chunks.at(4)==RangePair(16,21));
  }

  SUBCASE("Too short for last"){
    const auto chunks = generate_n_chunks(19, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==4);
    CHECK(chunks.at(0)==RangePair( 0, 4));
    CHECK(chunks.at(1)==RangePair( 4, 8));
    CHECK(chunks.at(2)==RangePair( 8,12));
    CHECK(chunks.at(3)==RangePair(12,16));
    CHECK(chunks.at(4)==RangePair(16,19));
  }

  SUBCASE("Too short for last and below 0.5 point"){
    const auto chunks = generate_n_chunks(17, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==5);
    CHECK(chunks.at(0)==RangePair( 0, 3));
    CHECK(chunks.at(1)==RangePair( 3, 6));
    CHECK(chunks.at(2)==RangePair( 6, 9));
    CHECK(chunks.at(3)==RangePair( 9,12));
    CHECK(chunks.at(4)==RangePair(12,17));
  }

  SUBCASE("Offset"){
    const auto chunks = generate_n_chunks(5, 22, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==5);
    CHECK(chunks.at(0)==RangePair( 5, 8));
    CHECK(chunks.at(1)==RangePair( 8,11));
    CHECK(chunks.at(2)==RangePair(11,14));
    CHECK(chunks.at(3)==RangePair(14,17));
    CHECK(chunks.at(4)==RangePair(17,22));
  }

  SUBCASE("Offset From Pair"){
    const RangePair range_pair(5,22);
    const auto chunks = generate_n_chunks(range_pair, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==5);
    CHECK(chunks.at(0)==RangePair( 5, 8));
    CHECK(chunks.at(1)==RangePair( 8,11));
    CHECK(chunks.at(2)==RangePair(11,14));
    CHECK(chunks.at(3)==RangePair(14,17));
    CHECK(chunks.at(4)==RangePair(17,22));
  }
}



TEST_CASE("generate_chunks_of_size"){
  SUBCASE("Zero size"){
    CHECK_THROWS_AS(generate_chunks_of_size(10,0), std::runtime_error);
  }

  SUBCASE("Even"){
    const auto chunks = generate_chunks_of_size(12, 3);
    CHECK(chunks.at(0)==RangePair(0, 3));
    CHECK(chunks.at(1)==RangePair(3, 6));
    CHECK(chunks.at(2)==RangePair(6, 9));
    CHECK(chunks.at(3)==RangePair(9,12));
  }

  SUBCASE("Short"){
    const auto chunks = generate_chunks_of_size(12, 5);
    CHECK(chunks.at(0)==RangePair( 0, 5));
    CHECK(chunks.at(1)==RangePair( 5,10));
    CHECK(chunks.at(2)==RangePair(10,12));
  }

  SUBCASE("Even"){
    const auto chunks = generate_chunks_of_size(RangePair(12,24), 3);
    CHECK(chunks.at(0)==RangePair(12,15));
    CHECK(chunks.at(1)==RangePair(15,18));
    CHECK(chunks.at(2)==RangePair(18,21));
    CHECK(chunks.at(3)==RangePair(21,24));
  }

  SUBCASE("Short"){
    const auto chunks = generate_chunks_of_size(RangePair(12,24), 5);
    CHECK(chunks.at(0)==RangePair(12,17));
    CHECK(chunks.at(1)==RangePair(17,22));
    CHECK(chunks.at(2)==RangePair(22,24));
  }
}