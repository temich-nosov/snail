#ifndef MAXPOOLLAYER_H
#define MAXPOOLLAYER_H

#include "DataArray.h"
#include "Layer.h"
#include "FileTools.h"

#include <algorithm>
#include <stdexcept>
 
/**
 * Класс слоя объединения
 */
class MaxPoolLayer : public Layer {
  DataArray::Size inputSize;
  DataArray::Size outputSize;
  int filterSize;

public:
  DataArray::Size getInputSize() const;
  DataArray::Size getOutputSize() const;

  MaxPoolLayer(DataArray::Size inputSize, int filterSize);

  virtual void propagate(const DataArray & input, DataArray & output);
  virtual void backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda);

  void write(std::ostream & stream) const;
  static MaxPoolLayer* read(std::istream & stream);

  ~MaxPoolLayer() {}

  virtual Layer::LayerType getType() const;
};

#endif
