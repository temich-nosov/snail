#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"

#include "ConvolutionalLayer.h"
#include "MaxPoolLayer.h"
#include "FullConnectedLayer.h"

#include "DataArray.h"
#include "FileTools.h"

#include <iostream>
#include <stdexcept>

class NeuralNetwork {
  std::vector<Layer*> layers;
  std::vector<DataArray> data;
  std::vector<DataArray> error;

  static Layer* readLayer(std::istream & stream);
  static void writeLayer(const Layer* layer, std::ostream & stream);
public:
  NeuralNetwork();
  
  /// Очистить сеть
  void clear();

  /// Загрузить сеть из файла
  void load(std::string filename);

  /// Сохранить сеть в файл
  void save(std::string filename) const;

  /// Добавить слой в сеть
  void addLayer(Layer * layer);

  /// Пропустить данные через сеть
  void propagate(const DataArray & input, DataArray & output);

  /// Произвести итерацию обучения сети
  void backPropagate(const DataArray & input, const DataArray & expectOutput, float lambda);

  ~NeuralNetwork();
};

#endif
