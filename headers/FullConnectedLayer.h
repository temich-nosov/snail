#pragma once

#include "DataArray.h"
#include "ConvolutionalLayer.h"
#include "FileTools.h"

namespace snail {
/**
 * Полносвязный слой
 */
class FullConnectedLayer : public ConvolutionalLayer {
  typedef ConvolutionalLayer Base;
  FullConnectedLayer(const ConvolutionalLayer & c);
public:
  FullConnectedLayer(DataArray::Size inputSize, int depth, float maxRnd);
  ~FullConnectedLayer() {}

  void write(std::ostream & stream) const;
  static FullConnectedLayer* read(std::istream & stream);

  Layer::LayerType getType() const;
};
} // namespace snail
