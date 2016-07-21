#include "MaxPoolLayer.h"


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


MaxPoolLayer::MaxPoolLayer(DataArray::Size inputSize, int filterSize) : inputSize(inputSize), filterSize(filterSize) {
  outputSize = inputSize;
  outputSize.w /= filterSize;
  outputSize.h /= filterSize;
}


DataArray::Size MaxPoolLayer::getInputSize() const {
  return inputSize;
}


DataArray::Size MaxPoolLayer::getOutputSize() const {
  return outputSize;
}
