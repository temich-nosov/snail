#include "DataArray.h"


namespace snail {
bool DataArray::Size::check(int x, int y, int z) const {
  return (x >= 0 && x < w) && (y >= 0 && y < h) && (z >= 0 && z < d);
}


bool DataArray::Size::operator==(const Size & size) const {
  return (w == size.w) && (h == size.h) && (d == size.d);
}


bool DataArray::Size::operator!=(const Size & size) const {
  return !((*this) == size);
}


DataArray::Size DataArray::Size::read(std::istream & stream) {
  DataArray::Size size;
  size.w = readInt(stream);
  size.h = readInt(stream);
  size.d = readInt(stream);
  return size;
}


void DataArray::Size::write(std::ostream & stream) const {
  writeInt(stream, w);
  writeInt(stream, h);
  writeInt(stream, d);
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


DataArray::DataArray(const DataArray & dataArray) {
  size = dataArray.size;
  arr = new float[size.elemCnt()];
  std::copy(dataArray.arr, dataArray.arr + dataArray.size.elemCnt(), arr);
}


DataArray::~DataArray() {
  if (arr)
    delete[] arr;
}


DataArray& DataArray::operator=(const DataArray & other) {
  if (other.arr == 0) {
    if (arr)
      delete[] arr;
    size = other.size;

    return *this;
  }

  if (size != other.size) {
    if (arr)
      delete[] arr;

    arr = new float[other.size.elemCnt()];
    size = other.size;
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


void DataArray::fillRnd(float minVal, float maxVal) {
  std::generate_n(arr, size.elemCnt(), [minVal, maxVal] ()->float {
    return minVal + rand()/(RAND_MAX / (maxVal-minVal));
  });
}


void DataArray::addZeros(int width, DataArray & output) const {
  Size newSize = size;
  newSize.w += 2 * width;
  newSize.h += 2 * width;

  if (newSize != output.getSize())
    output = DataArray(newSize);

  output.clear();
  for (int z = 0; z < getSize().d; ++z) {
    for (int y = 0; y < getSize().h; ++y) {
      for (int x = 0; x < getSize().w; ++x) {
        output.at(x + width, y + width, z) = at(x, y, z);
      }
    }
  }
}


void DataArray::removeFrame(int width, DataArray & output) const {
  Size newSize = size;
  newSize.w -= 2 * width;
  newSize.h -= 2 * width;

  if (newSize.w <= 0 || newSize.h <= 0)
    throw std::invalid_argument("DataArray::removeFrame() new size less or equal to zero");

  if (newSize != output.getSize())
    output = DataArray(newSize);

  for (int z = 0; z < newSize.d; ++z)
    for (int y = 0; y < newSize.h; ++y)
      for (int x = 0; x < newSize.w; ++x)
        output.at(x, y, z) = at(x + width, y + width, z);
}


DataArray DataArray::read(std::istream & stream) {
  DataArray::Size size = DataArray::Size::read(stream);
  DataArray res(size);

  for (int z = 0; z < size.d; ++z)
    for (int y = 0; y < size.h; ++y)
      for (int x = 0; x < size.w; ++x)
        res.at(x, y, z) = readFloat(stream);

  return res;
}


void DataArray::write(std::ostream & stream) const {
  size.write(stream);

  for (int z = 0; z < size.d; ++z)
    for (int y = 0; y < size.h; ++y)
      for (int x = 0; x < size.w; ++x)
        writeFloat(stream, at(x, y, z));
}


void swap( DataArray& a, DataArray& b ) {
  std::swap(a.arr, b.arr);
  std::swap(a.size, b.size);
}
} // namespace snail
