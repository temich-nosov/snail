#ifndef FULLCONNECTEDLAYER_H
#define FULLCONNECTEDLAYER_H

/**
 * Полносвязный слой
 */
class FullConnectedLayer : public ConvolutionalLayer {
public:
  FullConnectedLayer();
  ~FullConnectedLayer();
};

/**
 * Слой активации
 */
class ReluLayer : public Layer {
  DataArray::Size inputOutputSize;
public:
  DataArray::Size getInputSize() const;
  DataArray::Size getOutputSize() const;

  ReluLayer(DataArray::Size size) : inputOutputSize(size) {}

  virtual void propagate(const DataArray & input, DataArray & output);
  virtual void backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda);

  ~ReluLayer() {}
};

#endif
