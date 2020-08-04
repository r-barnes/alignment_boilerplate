#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <stack>
#include <stdexcept>
#include <vector>

namespace albp {

///@brief Get the sorted positions of elements in a vector
template<class OrderingFunction>
std::vector<uint32_t> sorted_ordering(const size_t N, OrderingFunction func){
  std::vector<uint32_t> ordering(N);
  std::iota(ordering.begin(), ordering.end(), 0);
  std::sort(ordering.begin(), ordering.end(), func);
  return ordering;
}



///@brief Reorder a vector by moving its elements to indices indicted by another
///       vector. Takes O(N) time and O(N) space. Allocations are amoritzed.
///
///@param[in,out] values   Vector to be reordered
///@param[in]     ordering A permutation of the vector
template<class ValueType, class OrderingType>
void forward_reorder(
  std::vector<ValueType>          &values,
  const std::vector<OrderingType> &ordering
){
  thread_local std::vector<uint16_t> visited;

  if(ordering.size()!=values.size()){
    throw std::runtime_error("ordering and values must be the same size!");
  }

  //Size the visited vector appropriately. Since vectors don't shrink, this will
  //shortly become large enough to handle most of the inputs. The vector is 1
  //larger than necessary because the first element is special.
  if(visited.empty() || visited.size()-1<values.size()){
    visited.resize(values.size()+1);
  }

  //If the visitation indicator becomes too large, we reset everything. This is
  //O(N) expensive, but unlikely to occur in most use cases if an appropriate
  //data type is chosen for the visited vector. For instance, an unsigned 32-bit
  //integer provides ~4B uses before it needs to be reset. We subtract one below
  //to avoid having to think too much about off-by-one errors. Note that
  //choosing the biggest data type possible is not necessarily a good idea!
  //Smaller data types will have better cache utilization.
  if(visited.at(0)==std::numeric_limits<uint16_t>::max()-1)
    std::fill(visited.begin(), visited.end(), 0);

  //We increment the stored visited indicator and make a note of the result. Any
  //value in the visited vector less than `visited_indicator` has not been
  //visited.
  const auto visited_indicator = ++visited.at(0);

  //For doing an early exit if we get everything in place
  auto remaining = values.size();

  //For all elements that need to be placed
  for(size_t s=0;s<ordering.size() && remaining>0;s++){
    assert(visited[s+1]<=visited_indicator);

    //Ignore already-visited elements
    if(visited[s+1]==visited_indicator)
      continue;

    //Don't rearrange if we don't have to
    if(s==visited[s])
      continue;

    //Follow this cycle, putting elements in their places until we get back
    //around. Use move semantics for speed.
    auto temp = std::move(values[s]);
    auto i = s;
    for(;s!=(size_t)ordering[i];i=ordering[i],--remaining){
      std::swap(temp, values[ordering[i]]);
      visited[i+1] = visited_indicator;
    }
    std::swap(temp, values[s]);
    visited[i+1] = visited_indicator;
  }
}



///@brief Reorder a vector by moving its elements to indices indicted by another
///       vector. Takes O(2N) time and O(2N) space. Allocations are amoritzed.
///
///@param[in,out] values   Vector to be reordered
///@param[in]     ordering A permutation of the vector
template<class ValueType, class OrderingType>
void backward_reorder(
  ValueType *const values,
  const std::vector<OrderingType> &ordering
){
  //The orderings form a linked list. We need O(N) memory to reverse a linked
  //list. We use `thread_local` so that the function is reentrant.
  thread_local std::stack<OrderingType> stack;
  thread_local std::vector<uint16_t> visited;

  //Size the visited vector appropriately. Since vectors don't shrink, this will
  //shortly become large enough to handle most of the inputs. The vector is 1
  //larger than necessary because the first element is special.
  if(visited.empty() || visited.size()-1<ordering.size()){
    visited.resize(ordering.size()+1);
  }

  //If the visitation indicator becomes too large, we reset everything. This is
  //O(N) expensive, but unlikely to occur in most use cases if an appropriate
  //data type is chosen for the visited vector. For instance, an unsigned 32-bit
  //integer provides ~4B uses before it needs to be reset. We subtract one below
  //to avoid having to think too much about off-by-one errors. Note that
  //choosing the biggest data type possible is not necessarily a good idea!
  //Smaller data types will have better cache utilization.
  if(visited.at(0)==std::numeric_limits<uint16_t>::max()-1)
    std::fill(visited.begin(), visited.end(), 0);

  //We increment the stored visited indicator and make a note of the result. Any
  //value in the visited vector less than `visited_indicator` has not been
  //visited.
  const auto visited_indicator = ++visited.at(0);

  //For doing an early exit if we get everything in place
  auto remaining = ordering.size();

  //For all elements that need to be placed
  for(size_t s=0;s<ordering.size() && remaining>0;s++){
    assert(visited[s+1]<=visited_indicator);

    //Ignore already-visited elements
    if(visited[s+1]==visited_indicator)
      continue;

    //Don't rearrange if we don't have to
    if(s==visited[s])
      continue;

    //The orderings form a linked list. We need to follow that list to its end
    //in order to reverse it.
    stack.emplace(s);
    for(auto i=s;s!=(size_t)ordering[i];i=ordering[i]){
      stack.emplace(ordering[i]);
    }

    //Now we follow the linked list in reverse to its beginning, putting
    //elements in their places. Use move semantics for speed.
    auto temp = std::move(values[s]);
    while(!stack.empty()){
      std::swap(temp, values[stack.top()]);
      visited[stack.top()+1] = visited_indicator;
      stack.pop();
      --remaining;
    }
    visited[s+1] = visited_indicator;
  }
}

}