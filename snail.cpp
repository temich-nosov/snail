#include <iostream>
#include <algorithm>
#include "snail.h"



bool DataArray::Size::check(int x, int y, int z) const {
  return (x >= 0 && x < w) && (y >= 0 && y < h) && (z >= 0 && z < d);
}


bool DataArray::Size::operator==(const Size & size) const {
  return (w == size.w) && (h == size.h) && (d == size.d);
}

bool DataArray::Size::operator!=(const Size & size) const {
  return !((*this) == size);
}


DataArray::DataArray() {
  arr = 0;
}


DataArray::DataArray(Size size) : size(size) {
  arr = new float[size.elemCnt()];
}


DataArray::DataArray(int w, int h, int d) : size(w, h, d) {
  arr = new float[size.elemCnt()];
}


DataArray::~DataArray() {
  if (arr)
    delete[] arr;
}


DataArray& DataArray::operator=(const DataArray & other) {
  if (size != other.size) {
    // Do smth
    std::cerr << "Размеры не совпадают" << std::endl;
    return *this;
  }

  std::copy(other.arr, other.arr + other.size.elemCnt(), arr);
  return *this;
}


float & DataArray::at(int x, int y, int z) {
  return arr[z * size.w * size.h + y * size.w + x];
}


const float & DataArray::at(int x, int y, int z) const {
  return arr[z * size.w * size.h + y * size.w + x];
}


void DataArray::clear() {
  std::fill(arr, arr + size.elemCnt(), 0);
}
