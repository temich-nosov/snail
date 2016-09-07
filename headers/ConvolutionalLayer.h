#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include "DataArray.h"
#include "Layer.h"
#include "functions.h"
#include "FileTools.h"

#include <vector>
#include <utility>
#include <iostream>
#include <stdexcept>


namespace snail {
/**
 * Класс свёрточного слоя
 */
class ConvolutionalLayer : public Layer {
  DataArray realInput;
  DataArray realInputError;

  DataArray::Size inputSize;
  DataArray::Size realInputSize;
  DataArray::Size outputSize;

  int depth;       /// Число фильтров в слое
  int stride;      /// Шаг
  int zeroPadding; /// Дополнение нулями
  int filterSize;  /// Размер части, выбираемой фильтром

  std::vector< std::pair<DataArray, float> > filters; // массив весов связей и смещение
  ConvolutionalLayer();

public:
  ConvolutionalLayer(DataArray::Size inputSize, int depth, int stride, int zeroPadding, int filterSize, float maxRnd);

  DataArray::Size getInputSize() const;
  DataArray::Size getOutputSize() const;

  virtual void propagate(const DataArray & input, DataArray & output);
  virtual void backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda);

  void write(std::ostream & stream) const;
  static ConvolutionalLayer* read(std::istream & stream);

  ~ConvolutionalLayer() {}

  virtual Layer::LayerType getType() const;
};
} // namespace snail

#endif
