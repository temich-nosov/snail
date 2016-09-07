#include "NeuralNetwork.h"


namespace snail {
void NeuralNetwork::addLayer(Layer * layer) {
  if (!layers.empty() && layers.back()->getOutputSize() != layer->getInputSize())
    throw std::invalid_argument("NeuralNetwork::addLayer() Input size of the new layer is not equal to the output size of the previous layer");

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
}

/// Пропустить данные через сеть
void NeuralNetwork::propagate(const DataArray & input, DataArray & output) {
  if (layers.empty())
    throw std::invalid_argument("NeuralNetwork::propagate() void neural network");

  if (input.getSize() != layers[0]->getInputSize())
    throw std::invalid_argument("NeuralNetwork::propagate() bad input size");

  if (output.getSize() != layers.back()->getOutputSize())
    throw std::invalid_argument("NeuralNetwork::propagate() bad output size");

  data[0] = input;
  for (int i = 0; i < (int)layers.size(); ++i)
    layers[i]->propagate(data[i], data[i + 1]);

  output = data.back();
}

/// Произвести итерацию обучения сети
void NeuralNetwork::backPropagate(const DataArray & input, const DataArray & expectOutput, float lambda) {
  if (layers.empty())
    throw std::invalid_argument("NeuralNetwork::backPropagate() void neural network");

  if (input.getSize() != layers[0]->getInputSize())
    throw std::invalid_argument("NeuralNetwork::backPropagate() bad input size");

  if (expectOutput.getSize() != layers.back()->getOutputSize())
    throw std::invalid_argument("NeuralNetwork::backPropagate() bad output size");

  propagate(input, error.back());
  for (int z = 0; z < error.back().getSize().d; ++z)
    for (int y = 0; y < error.back().getSize().h; ++y)
      for (int x = 0; x < error.back().getSize().w; ++x)
        error.back().at(x, y, z) -= expectOutput.at(x, y, z);

  for (int i = layers.size() - 1; i >= 0; --i) {
    error[i].clear();
    layers[i]->backPropagate(data[i], data[i + 1], error[i + 1], error[i], lambda);
  }
}


NeuralNetwork::NeuralNetwork() {}


void NeuralNetwork::clear() {
  for (auto t : layers)
    delete t;
  layers.clear();
  data.clear();
  error.clear();
}


NeuralNetwork::~NeuralNetwork() {
  clear();
}


void NeuralNetwork::load(std::string filename) {
  clear();
  std::ifstream file(filename, std::ifstream::binary);
  int cnt;
  cnt = readInt(file);

  for (int i = 0; i < cnt; ++i)
    addLayer(readLayer(file));
}


void NeuralNetwork::save(std::string filename) const {
  std::ofstream file(filename, std::ofstream::binary);
  writeInt(file, layers.size());

  for (const Layer* t : layers)
    writeLayer(t, file);
}


Layer* NeuralNetwork::readLayer(std::istream & stream) {
  Layer::LayerType layerType = static_cast<Layer::LayerType>(readInt(stream));
  if (layerType == Layer::LayerType::CONVOLUTIONAL)
    return ConvolutionalLayer::read(stream);
  if (layerType == Layer::LayerType::MAX_POOL)
    return MaxPoolLayer::read(stream);

  // layerType == FULL_CONNECTED
  return FullConnectedLayer::read(stream);
}

void NeuralNetwork::writeLayer(const Layer* layer, std::ostream & stream) {
  int type = layer->getType();
  writeInt(stream, type);
  layer->write(stream);
}
} // namespace snail
