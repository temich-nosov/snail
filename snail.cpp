#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include "snail.h"

float derivative(float x) {
  float f = func(x);
  return f * (1.f - f);
}

float func(float x) {
  return 1.f / (1.f + exp(-x));
}

float softPlusFunction(float x) {
  return log(1.f + exp(x));
}

float derivativeSoftPlusFunction(float x) {
  return 1.f / (1.f + exp(-x));
}

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
    // Размеры не совпадают    
    
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

  if (newSize.w <= 0 || newSize.h <= 0) {
    std::cerr << "..." << std::endl;
    return;
  }

  if (newSize != output.getSize())
    output = DataArray(newSize);

  for (int z = 0; z < newSize.d; ++z) {
    for (int y = 0; y < newSize.h; ++y) {
      for (int x = 0; x < newSize.w; ++x) {
        output.at(x, y, z) = at(x + width, y + width, z);
      }
    }
  }
}


Layer::~Layer() {}


ConvolutionalLayer::ConvolutionalLayer(DataArray::Size inputSize, int depth, int stride, int zeroPadding, int filterSize) :
  inputSize(inputSize), depth(depth), stride(stride), zeroPadding(zeroPadding), filterSize(filterSize) {

  realInputSize = inputSize;

  realInputSize.w += 2 * zeroPadding;
  realInputSize.h += 2 * zeroPadding;

  realInput = DataArray(realInputSize);
  realInputError = DataArray(realInputSize);

  int inW = realInputSize.w - filterSize;
  int inH = realInputSize.h - filterSize;

  if (inW < 0 || inW % stride != 0) std::cerr << "Неправильные параметры" << std::endl; // error
  if (inH < 0 || inH % stride != 0) std::cerr << "Неправильные параметры" << std::endl; // error

  outputSize.w = inW / stride + 1;
  outputSize.h = inH / stride + 1;
  outputSize.d = depth;

  for (int i = 0; i < depth; ++i) {
    filters.push_back( std::pair<DataArray, float>(DataArray(filterSize, filterSize, inputSize.d), 0) );
    filters.back().first.fillRnd(0, 0.01);
    // filters.back().first.clear();
  }
}


void ConvolutionalLayer::propagate(const DataArray & input, DataArray & output) {
  if (input.getSize() != inputSize || output.getSize() != outputSize) {
    std::cerr << "Неправильные размеры входных или выходных данных" << std::endl;
    return;
  }
  
  input.addZeros(zeroPadding, realInput);

  for (int d = 0; d < depth; ++d) {
    const DataArray & filter = filters[d].first;

    for (int xo = 0; xo < outputSize.w; ++xo) {
      for (int yo = 0; yo < outputSize.h; ++yo) {
        float & val = output.at(xo, yo, d);
        val = 0;

        int ymin = yo * stride;
        int ymax = ymin + filterSize;

        int xmin = xo * stride;
        int xmax = xmin + filterSize;

        for (int z = 0; z < inputSize.d; ++z) {
          for (int y = ymin; y < ymax; ++y) {
            for (int x = xmin; x < xmax; ++x) {
              val += realInput.at(x, y, z) * filter.at(x - xmin, y - ymin, z);
            }
          }
        }

        val = func(val + filters[d].second);
      }
    }
  }
}


void ConvolutionalLayer::backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda) {
  if (inputError.getSize() != getInputSize() || input.getSize() != getInputSize()) {
    std::cerr << "Ошибка" << std::endl;
    return;
  }

  if (error.getSize() != getOutputSize() || output.getSize() != getOutputSize()) {
    std::cerr << "Ошибка" << std::endl;
    return;
  }

  inputError.addZeros(zeroPadding, realInputError);
  input.addZeros(zeroPadding, realInput);

  for (int d = 0; d < depth; ++d) {
    DataArray & filter = filters[d].first;

    for (int xo = 0; xo < outputSize.w; ++xo) {
      for (int yo = 0; yo < outputSize.h; ++yo) {
        float err = error.at(xo, yo, d);
        float dr = output.at(xo, yo, d);
        float cf = err * (1 - dr) * dr;

        int ymin = yo * stride;
        int ymax = ymin + filterSize;

        int xmin = xo * stride;
        int xmax = xmin + filterSize;

        for (int z = 0; z < inputSize.d; ++z) {
          for (int y = ymin; y < ymax; ++y) {
            for (int x = xmin; x < xmax; ++x) {
              realInputError.at(x, y, z) += cf * filter.at(x - xmin, y - ymin, z);
            }
          }
        }
      }
    }
  }

  for (int d = 0; d < depth; ++d) {
    DataArray & filter = filters[d].first;

    for (int xo = 0; xo < outputSize.w; ++xo) {
      for (int yo = 0; yo < outputSize.h; ++yo) {
        float err = error.at(xo, yo, d);
        float dr = output.at(xo, yo, d);
        float cf = err * lambda * (1 - dr) * dr;

        // std::cout << err << " " << dr << " " << cf << std::endl;

        int ymin = yo * stride;
        int ymax = yo * stride + filterSize;

        int xmin = xo * stride;
        int xmax = xo * stride + filterSize;

        for (int z = 0; z < inputSize.d; ++z) {
          for (int y = ymin; y < ymax; ++y) {
            for (int x = xmin; x < xmax; ++x) {
              // if (z == 1 && y - ymin == 0 && x - xmin == 0) { std::cout << cf << " " << realInput.at(x, y, z) << std::endl; }
              filter.at(x - xmin, y - ymin, z) -= cf * realInput.at(x, y, z);
            }
          }
        }

        filters[d].second -= cf;
      }
    }
  }

  realInputError.removeFrame(zeroPadding, inputError);
}



void MaxPoolLayer::propagate(const DataArray & input, DataArray & output) {
  output.clear();
  // TODO Прочекать размеры и так далее
  for (int z = 0; z < outputSize.d; ++z) {
    for (int y = 0; y < outputSize.h; y += filterSize) {
      for (int x = 0; x < outputSize.w; x += filterSize) {
        float & res = output.at(x / filterSize, y / filterSize, z);
        res = input.at(x, y, z);
        for (int dy = 0; dy < filterSize; ++dy) {
          for (int dx = 0; dx < filterSize; ++dx) {
            res = std::max(res, input.at(x + dx, y + dy, z));
          }
        }
      }
    }
  }
}


void MaxPoolLayer::backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda) {
  inputError.clear();
  // TODO Прочекать размеры и так далее
  for (int z = 0; z < outputSize.d; ++z) {
    for (int y = 0; y < outputSize.h; y += filterSize) {
      for (int x = 0; x < outputSize.w; x += filterSize) {
        int ddx = 0, ddy = 0;
        float res = input.at(x, y, z);
        for (int dy = 0; dy < filterSize; ++dy) {
          for (int dx = 0; dx < filterSize; ++dx) {
            if (input.at(x + dx, y + dy, z) > res) {
              res = input.at(x + dx, y + dy, z);
              ddx = dx;
              ddy = dy;
            }
          }
        }

        inputError.at(x + ddx, y + ddy, z) = error.at(x / filterSize, y / filterSize, z);
      }
    }
  }
}


DataArray::Size ReluLayer::getInputSize() const {
  return inputOutputSize;
}

DataArray::Size ReluLayer::getOutputSize() const {
  return inputOutputSize;
}

void ReluLayer::propagate(const DataArray & input, DataArray & output) {
  for (int z = 0; z < inputOutputSize.d; ++z) {
    for (int y = 0; y < inputOutputSize.h; ++y) {
      for (int x = 0; x < inputOutputSize.w; ++x) {
        output.at(x, y, z) = softPlusFunction(input.at(x, y, z));
      }
    }
  }
}

void ReluLayer::backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda) {
  for (int z = 0; z < inputOutputSize.d; ++z) {
    for (int y = 0; y < inputOutputSize.h; ++y) {
      for (int x = 0; x < inputOutputSize.w; ++x) {
        inputError.at(x, y, z) = error.at(x, y, z) * derivativeSoftPlusFunction(input.at(x, y, z));
      }
    }
  }
}

bool NeuralNetwork::addLayer(Layer * layer) {
  if (!layers.empty() && layers.back()->getOutputSize() != layer->getInputSize())
    return false;

  if (layers.empty()) {
    data.push_back(DataArray(layer->getInputSize()));
    data[0].clear();

    error.push_back(DataArray(layer->getInputSize()));
    error[0].clear();
  }

  data.push_back(DataArray(layer->getOutputSize()));
  data.back().clear();

  error.push_back(DataArray(layer->getOutputSize()));
  error.back().clear();

  layers.push_back(layer);
  
  return true;
}

/// Пропустить данные через сеть
void NeuralNetwork::propagate(const DataArray & input, DataArray & output) {
  if (layers.empty()) {
    std::cerr << "Bad layers size" << std::endl;
    return;
  }

  if (input.getSize() != layers[0]->getInputSize()) {
    std::cerr << "Bad input size" << std::endl;
    return;
  }

  if (output.getSize() != layers.back()->getOutputSize()) {
    std::cerr << "Bad output size" << std::endl;
    return;
  }

  data[0] = input;
  for (int i = 0; i < layers.size(); ++i) {
    layers[i]->propagate(data[i], data[i + 1]);
  }

  output = data.back();
}

/// Произвести итерацию обучения сети
void NeuralNetwork::backPropagate(const DataArray & input, const DataArray & expectOutput, float lambda) {
  // TODO layers.size() != 0
  // TODO почекать размеры
  propagate(input, error.back()); // ...
  for (int z = 0; z < error.back().getSize().d; ++z) {
    for (int y = 0; y < error.back().getSize().h; ++y) {
      for (int x = 0; x < error.back().getSize().w; ++x) {
        error.back().at(x, y, z) -= expectOutput.at(x, y, z);
      }
    }
  }

  // Тут что то вычисляем в зависимости от используемой функции ошибки
  for (int i = layers.size() - 1; i >= 0; --i) {
    error[i].clear();
    layers[i]->backPropagate(data[i], data[i + 1], error[i + 1], error[i], lambda);
  }
}

NeuralNetwork::NeuralNetwork() {
}

void swap( DataArray& a, DataArray& b ) {
  // std::cerr << "YES!!!" << std::endl;
  std::swap(a.arr, b.arr);
  std::swap(a.size, b.size);
}
