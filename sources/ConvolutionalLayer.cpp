#include "ConvolutionalLayer.h"

DataArray::Size ConvolutionalLayer::getInputSize() const { return inputSize; }

DataArray::Size ConvolutionalLayer::getOutputSize() const { return outputSize; }

ConvolutionalLayer::ConvolutionalLayer() {}

ConvolutionalLayer::ConvolutionalLayer(DataArray::Size inputSize, int depth,
                                       int stride, int zeroPadding,
                                       int filterSize)
    : inputSize(inputSize),
      depth(depth),
      stride(stride),
      zeroPadding(zeroPadding),
      filterSize(filterSize) {
  realInputSize = inputSize;

  realInputSize.w += 2 * zeroPadding;
  realInputSize.h += 2 * zeroPadding;

  realInput = DataArray(realInputSize);
  realInputError = DataArray(realInputSize);

  int inW = realInputSize.w - filterSize;
  int inH = realInputSize.h - filterSize;

  if (inW < 0 || inW % stride != 0)
    throw std::invalid_argument(
        "ConvolutionalLayer::ConvolutionalLayer() wrong input width");
  if (inH < 0 || inH % stride != 0)
    throw std::invalid_argument(
        "ConvolutionalLayer::ConvolutionalLayer() wrong input height");

  outputSize.w = inW / stride + 1;
  outputSize.h = inH / stride + 1;
  outputSize.d = depth;

  for (int i = 0; i < depth; ++i) {
    filters.push_back(std::pair<DataArray, float>(
        DataArray(filterSize, filterSize, inputSize.d), 0));
    filters.back().first.fillRnd(0, 0.01);
  }
}

void ConvolutionalLayer::propagate(const DataArray& input, DataArray& output) {
  if (input.getSize() != inputSize)
    throw std::invalid_argument(
        "ConvolutionalLayer::propagate() Wrong input size");

  if (output.getSize() != outputSize)
    throw std::invalid_argument(
        "ConvolutionalLayer::propagate() Wrong output size");

  input.addZeros(zeroPadding, realInput);

  for (int d = 0; d < depth; ++d) {
    const DataArray& filter = filters[d].first;

    for (int xo = 0; xo < outputSize.w; ++xo) {
      for (int yo = 0; yo < outputSize.h; ++yo) {
        float& val = output.at(xo, yo, d);
        val = 0;

        int ymin = yo * stride;
        int ymax = ymin + filterSize;

        int xmin = xo * stride;
        int xmax = xmin + filterSize;

        for (int z = 0; z < inputSize.d; ++z)
          for (int y = ymin; y < ymax; ++y)
            for (int x = xmin; x < xmax; ++x)
              val += realInput.at(x, y, z) * filter.at(x - xmin, y - ymin, z);

        val = func(val + filters[d].second);
      }
    }
  }
}

void ConvolutionalLayer::backPropagate(const DataArray& input,
                                       const DataArray& output,
                                       const DataArray& error,
                                       DataArray& inputError, float lambda) {
  if (inputError.getSize() != getInputSize() ||
      input.getSize() != getInputSize())
    throw std::invalid_argument(
        "ConvolutionalLayer::propagate() Wrong input size");

  if (error.getSize() != getOutputSize() || output.getSize() != getOutputSize())
    throw std::invalid_argument(
        "ConvolutionalLayer::propagate() Wrong output size");

  inputError.addZeros(zeroPadding, realInputError);
  input.addZeros(zeroPadding, realInput);

  for (int d = 0; d < depth; ++d) {
    DataArray& filter = filters[d].first;

    for (int xo = 0; xo < outputSize.w; ++xo) {
      for (int yo = 0; yo < outputSize.h; ++yo) {
        float err = error.at(xo, yo, d);
        float dr = output.at(xo, yo, d);
        float cf = err * (1 - dr) * dr;

        int ymin = yo * stride;
        int ymax = ymin + filterSize;

        int xmin = xo * stride;
        int xmax = xmin + filterSize;

        for (int z = 0; z < inputSize.d; ++z)
          for (int y = ymin; y < ymax; ++y)
            for (int x = xmin; x < xmax; ++x)
              realInputError.at(x, y, z) += cf * filter.at(x - xmin, y - ymin, z);
      }
    }
  }

  for (int d = 0; d < depth; ++d) {
    DataArray& filter = filters[d].first;

    for (int xo = 0; xo < outputSize.w; ++xo) {
      for (int yo = 0; yo < outputSize.h; ++yo) {
        float err = error.at(xo, yo, d);
        float dr = output.at(xo, yo, d);
        float cf = err * lambda * (1 - dr) * dr;

        int ymin = yo * stride;
        int ymax = yo * stride + filterSize;

        int xmin = xo * stride;
        int xmax = xo * stride + filterSize;

        for (int z = 0; z < inputSize.d; ++z)
          for (int y = ymin; y < ymax; ++y)
            for (int x = xmin; x < xmax; ++x)
              filter.at(x - xmin, y - ymin, z) -= cf * realInput.at(x, y, z);

        filters[d].second -= cf;
      }
    }
  }

  realInputError.removeFrame(zeroPadding, inputError);
}


void ConvolutionalLayer::write(std::ostream & stream) const {
  inputSize.write(stream);
  writeInt(stream, depth);
  writeInt(stream, stride);
  writeInt(stream, zeroPadding);
  writeInt(stream, filterSize);

  for (auto & f : filters) {
    f.first.write(stream);
    writeFloat(stream, f.second);
  }
}

ConvolutionalLayer* ConvolutionalLayer::read(std::istream& stream) {
  ConvolutionalLayer* res = new ConvolutionalLayer();

  res->inputSize = DataArray::Size::read(stream);
  res->depth = readInt(stream);
  res->stride = readInt(stream);
  res->zeroPadding = readInt(stream);
  res->filterSize = readInt(stream);

  res->realInputSize = res->inputSize;

  res->realInputSize.w += 2 * res->zeroPadding;
  res->realInputSize.h += 2 * res->zeroPadding;

  res->realInput = DataArray(res->realInputSize);
  res->realInputError = DataArray(res->realInputSize);

  int inW = res->realInputSize.w - res->filterSize;
  int inH = res->realInputSize.h - res->filterSize;

  if (inW < 0 || inW % res->stride != 0)
    throw std::invalid_argument(
        "ConvolutionalLayer::ConvolutionalLayer() wrong input width");
  if (inH < 0 || inH % res->stride != 0)
    throw std::invalid_argument(
        "ConvolutionalLayer::ConvolutionalLayer() wrong input height");

  res->outputSize.w = inW / res->stride + 1;
  res->outputSize.h = inH / res->stride + 1;
  res->outputSize.d = res->depth;
  res->filters.resize(res->depth);

  for (int i = 0; i < res->depth; ++i) {
    res->filters[i].first = DataArray::read(stream);
    res->filters[i].second = readFloat(stream);
  }

  return res;
}


Layer::LayerType ConvolutionalLayer::getType() const {
  return Layer::LayerType::CONVOLUTIONAL;
}
