#include "MaxPoolLayer.h"


namespace snail {
void MaxPoolLayer::propagate(const DataArray & input, DataArray & output) {
  if (input.getSize() != inputSize)
    throw std::invalid_argument("MaxPoolLayer::propagate() bad input size");

  if (output.getSize() != outputSize)
    throw std::invalid_argument("MaxPoolLayer::propagate() bad output size");

  output.clear();

  for (int z = 0; z < outputSize.d; ++z) {
    for (int y = 0; y < outputSize.h; y += filterSize) {
      for (int x = 0; x < outputSize.w; x += filterSize) {
        float & res = output.at(x / filterSize, y / filterSize, z);
        res = input.at(x, y, z);
        for (int dy = 0; dy < filterSize; ++dy)
          for (int dx = 0; dx < filterSize; ++dx)
            res = std::max(res, input.at(x + dx, y + dy, z));
      }
    }
  }
}


void MaxPoolLayer::backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda) {
  if (input.getSize() != inputSize)
    throw std::invalid_argument("MaxPoolLayer::backPropagate() bad input size");

  if (output.getSize() != outputSize)
    throw std::invalid_argument("MaxPoolLayer::backPropagate() bad output size");

  if (inputError.getSize() != inputSize)
    throw std::invalid_argument("MaxPoolLayer::backPropagate() bad inputError size");

  if (error.getSize() != outputSize)
    throw std::invalid_argument("MaxPoolLayer::backPropagate() bad error size");

  inputError.clear();
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
  if (inputSize.w % filterSize != 0 || inputSize.h % filterSize != 0)
    throw std::invalid_argument("MaxPoolLayer::MaxPoolLayer() bad input size");

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


void MaxPoolLayer::write(std::ostream & stream) const {
  inputSize.write(stream);
  writeInt(stream, filterSize);
}


MaxPoolLayer* MaxPoolLayer::read(std::istream & stream) {
  DataArray::Size inputSize = DataArray::Size::read(stream);
  int filterSize = readInt(stream);
  return new MaxPoolLayer(inputSize, filterSize);
}


Layer::LayerType MaxPoolLayer::getType() const {
  return Layer::LayerType::MAX_POOL;
}
} // namespace snail
