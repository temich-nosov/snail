#include "NeuralNetwork.h"


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
  for (int i = 0; i < (int)layers.size(); ++i) {
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

