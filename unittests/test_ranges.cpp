#include "doctest.h"

#include <albp/ranges.hpp>

using namespace albp;

TEST_CASE("Ranges"){
  SUBCASE("Zero Chunk"){
    const auto chunks = generate_chunks(20, 1);
    CHECK(chunks.size()==1);
    CHECK(get_maximum_range(chunks)==20);
    CHECK(chunks.at(0)==RangePair(0, 20));
  }

  SUBCASE("One Chunk"){
    const auto chunks = generate_chunks(20, 1);
    CHECK(chunks.size()==1);
    CHECK(get_maximum_range(chunks)==20);
    CHECK(chunks.at(0)==RangePair(0, 20));
  }

  SUBCASE("Even"){
    const auto chunks = generate_chunks(20, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==4);
    CHECK(chunks.at(0)==RangePair( 0, 4));
    CHECK(chunks.at(1)==RangePair( 4, 8));
    CHECK(chunks.at(2)==RangePair( 8,12));
    CHECK(chunks.at(3)==RangePair(12,16));
    CHECK(chunks.at(4)==RangePair(16,20));
  }

  SUBCASE("Too long for last"){
    const auto chunks = generate_chunks(21, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==5);
    CHECK(chunks.at(0)==RangePair( 0, 4));
    CHECK(chunks.at(1)==RangePair( 4, 8));
    CHECK(chunks.at(2)==RangePair( 8,12));
    CHECK(chunks.at(3)==RangePair(12,16));
    CHECK(chunks.at(4)==RangePair(16,21));
  }

  SUBCASE("Too short for last"){
    const auto chunks = generate_chunks(19, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==4);
    CHECK(chunks.at(0)==RangePair( 0, 4));
    CHECK(chunks.at(1)==RangePair( 4, 8));
    CHECK(chunks.at(2)==RangePair( 8,12));
    CHECK(chunks.at(3)==RangePair(12,16));
    CHECK(chunks.at(4)==RangePair(16,19));
  }

  SUBCASE("Too short for last and below 0.5 point"){
    const auto chunks = generate_chunks(17, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==5);
    CHECK(chunks.at(0)==RangePair( 0, 3));
    CHECK(chunks.at(1)==RangePair( 3, 6));
    CHECK(chunks.at(2)==RangePair( 6, 9));
    CHECK(chunks.at(3)==RangePair( 9,12));
    CHECK(chunks.at(4)==RangePair(12,17));
  }

  SUBCASE("Offset"){
    const auto chunks = generate_chunks(5, 22, 5);
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
    const auto chunks = generate_chunks(range_pair, 5);
    CHECK(chunks.size()==5);
    CHECK(get_maximum_range(chunks)==5);
    CHECK(chunks.at(0)==RangePair( 5, 8));
    CHECK(chunks.at(1)==RangePair( 8,11));
    CHECK(chunks.at(2)==RangePair(11,14));
    CHECK(chunks.at(3)==RangePair(14,17));
    CHECK(chunks.at(4)==RangePair(17,22));
  }
}