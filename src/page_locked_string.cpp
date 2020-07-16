#include <rhgb/error_handling.hpp>
#include <rhgb/memory.hpp>
#include <rhgb/page_locked_string.hpp>

#include <cstring>
#include <stdexcept>

namespace rhgb {

PageLockedString::PageLockedString(size_t capacity) : _str(PageLockedMalloc<char>(capacity)), _capacity(capacity) {}

PageLockedString::~PageLockedString(){
  if(_str)
    RCHECKCUDAERROR(cudaFreeHost(_str));
}

PageLockedString& PageLockedString::operator+=(const std::string &o){
  if(_size+o.size()>_capacity)
    throw std::runtime_error("Appending to the PageLockedString would go above its capacity!");
  memcpy(&_str[_size], o.c_str(), o.size());
  _size += o.size();
  return *this;
}

char*       PageLockedString::data()      const { return _str;  }
size_t      PageLockedString::size()      const { return _size; }
size_t      PageLockedString::size_left() const { return _capacity-_size; }
bool        PageLockedString::empty()     const { return _size==0; }
bool        PageLockedString::full()      const { return _size==_capacity; }
std::string PageLockedString::str()       const { return std::string(_str, _str+_size); }
size_t      PageLockedString::capacity()  const { return _capacity; }
void        PageLockedString::clear()           { _size=0; }

}