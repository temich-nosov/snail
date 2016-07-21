#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include "DataArray.h"

#include <iostream>
#include <stdexcept>

class NeuralNetwork {
  std::vector<Layer *> layers;
  std::vector<DataArray> data;
  std::vector<DataArray> error;
public:
  NeuralNetwork();
  
  /// Очистить сеть
  void clear();

  // Загрузить сеть из файла
  // bool load(std::string filename);

  // Сохранить сеть в файл
  // bool save(std::string filename);

  /// Добавить слой в сеть
  void addLayer(Layer * layer);

  /// Пропустить данные через сеть
  void propagate(const DataArray & input, DataArray & output);

  /// Произвести итерацию обучения сети
  void backPropagate(const DataArray & input, const DataArray & expectOutput, float lambda);
};

#endif
