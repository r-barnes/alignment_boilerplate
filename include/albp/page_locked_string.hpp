#pragma once

#include <albp/memory.hpp>

#include <string>

namespace albp {

class PageLockedString {
  public:
    PageLockedString(size_t capacity);

    PageLockedString& operator+=(const std::string &o);
    PageLockedString& operator+=(const char o);

    char&       operator[](const size_t i);
    const char& operator[](const size_t i) const;

    char* data()      const;
    size_t size()     const;
    size_t size_left()const;
    bool empty()      const;
    bool full()       const;
    std::string str() const;
    size_t capacity() const;
    void clear();

  private:
    cuda_unique_hptr<char> _str;
    const size_t _capacity = 0;
    size_t _size = 0;
};

}