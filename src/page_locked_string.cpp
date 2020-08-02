#include <albp/error_handling.hpp>
#include <albp/memory.hpp>
#include <albp/page_locked_string.hpp>

#include <cstring>
#include <stdexcept>

namespace albp {

PageLockedString::PageLockedString(size_t capacity) : _str(PageLockedMalloc<char>(capacity)), _capacity(capacity) {}

PageLockedString& PageLockedString::operator+=(const std::string &o){
  if(_size+o.size()>_capacity)
    throw std::runtime_error("Appending to the PageLockedString would go above its capacity!");
  memcpy(&_str[_size], o.c_str(), o.size());
  _size += o.size();
  return *this;
}

PageLockedString& PageLockedString::operator+=(const char o){
  if(_size+1>_capacity)
    throw std::runtime_error("Appending to the PageLockedString would go above its capacity!");
  _str[_size] = o;
  _size += 1;
  return *this;
}

char&       PageLockedString::operator[](const size_t i)       { return _str[i]; }
const char& PageLockedString::operator[](const size_t i) const { return _str[i]; }

char*       PageLockedString::data()      const { return _str.get();  }
size_t      PageLockedString::size()      const { return _size; }
size_t      PageLockedString::size_left() const { return _capacity-_size; }
bool        PageLockedString::empty()     const { return _size==0; }
bool        PageLockedString::full()      const { return _size==_capacity; }
std::string PageLockedString::str()       const { return std::string(_str.get(), _str.get()+_size); }
size_t      PageLockedString::capacity()  const { return _capacity; }
void        PageLockedString::clear()           { _size=0; }

}