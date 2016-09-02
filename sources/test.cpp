#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>
#include <string>
#include <fstream>

#include "snail.h"

int readInt(std::ifstream & file) {
  int res;
  char* ptr = (char*)(&res);
  file.read(ptr, 4);
  std::swap(ptr[0], ptr[3]);
  std::swap(ptr[1], ptr[2]);
  return res;
}

float ubyteToFloat(unsigned char byte) {
  return float(byte) / 255;
}

void readImage(DataArray & dataArray, std::ifstream & file, int rows, int cols) {
  unsigned char * data = new unsigned char[rows * cols];
  file.read((char *)(data), rows * cols);
  
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dataArray.at(i, j, 0) = ubyteToFloat(data[i * cols + j]);
    }
  }

  delete data;
}

void getMNIST(std::vector< std::pair<DataArray, int> > & base, std::string imagesFilename, std::string labelsFilename) {
  std::ifstream baseFile(imagesFilename, std::ios::in | std::ios::binary);
  std::ifstream lableFile(labelsFilename, std::ios::in | std::ios::binary);

  if (!baseFile.is_open() || !lableFile.is_open()) {
    std::cout << "Не удалось открыть файл" << std::endl;
    return;
  }

  int sign = readInt(baseFile);
  if (sign != 2051) {
    std::cout << "Подпись файла с изображениями неправильная" << std::endl;
    return;
  }

  sign = readInt(lableFile);
  if (sign != 2049) {
    std::cout << "Подпись файла с ожидаемыми результатами неправильная" << std::endl;
    return;
  }

  int cnt = readInt(baseFile);
  if (cnt != readInt(lableFile)) {
    std::cout << "Кол-во записей в файлах не совпадает" << std::endl;
    return;
  }

  int rows = readInt(baseFile);
  int cols = readInt(baseFile);

  unsigned char * labels = new unsigned char[cnt];
  lableFile.read((char *)labels, cnt);

  base.resize(cnt);
  for (int i = 0; i < cnt; ++i) {
    DataArray data(rows, cols, 1);
    readImage(data, baseFile, rows, cols);
    base[i].first = data;
    base[i].second = labels[i];
  }

  delete labels;
}

int getVal(const DataArray & data) {
  int res = 0;
  for (int i = 0; i < 10; ++i) {
    if (data.at(0, 0, i) > data.at(0, 0, res))
      res = i;  
  }

  return res;
}

void genOut(int val, DataArray & data) {
  for (int i = 0; i < 10; ++i)
    data.at(0, 0, i) = ((i == val) ? 1.f : 0.f);
}

float sqrt(float x) {
  return x * x;
}

float sqrtDelta(const DataArray& a, const DataArray& b) {
  float res = 0;
  for (int i = 0; i < a.getSize().d; ++i)
    res += sqrt(a.at(0, 0, i) - b.at(0, 0, i));
  return res;
}

std::pair<float, float> check(NeuralNetwork & neuralNetwork, std::vector< std::pair<DataArray, int> > & data) {
  DataArray output(1, 1, 10);
  DataArray output2(1, 1, 10);
  int cnt = 0;
  float res = 0;

  for (auto& it : data) {
    neuralNetwork.propagate(it.first, output);
    genOut(it.second, output2);
    res += sqrtDelta(output, output2);
    if (getVal(output) == it.second)
      ++cnt;
  }

  return { res, float(cnt)/data.size() };
}

void teach(NeuralNetwork & neuralNetwork, std::vector< std::pair<DataArray, int> >& data, float step) {
  DataArray output(1, 1, 10);
  for (auto& it : data) {
    genOut(it.second, output);
    neuralNetwork.backPropagate(it.first, output, step);
  }
}

using std::cerr;
using std::endl;

int main(int argc, char ** argv) {
  if (argc != 3) {
    std::cout << "Необходимо передать имя файла с базой MNIST в качестве параметра" << std::endl;
    return 0;
  }

  std::vector< std::pair<DataArray, int> > base;
  getMNIST(base, argv[1], argv[2]);

  const int N = 1000;

  std::random_shuffle(base.begin(), base.end());
  std::vector< std::pair<DataArray, int> > checkBase(N);
  for (int i = 0; i < N; ++i)
    checkBase[i] = *(base.end() - i - 1);
  base.erase(base.end() - N, base.end());


  NeuralNetwork nn;
  nn.addLayer(new FullConnectedLayer(DataArray::Size(28, 28, 1), 10, 0.05));

  float learnStep = 1;
  for (int i = 0; i <= 30; ++i) {
    if (i == 5) learnStep = 0.1;
    if (i == 10) learnStep = 0.01;
    if (i == 15) learnStep = 0.001;

    auto pc = check(nn, checkBase);
    std::cout << "Iteration  : " << i  << std::endl;
    std::cout << "Match      : " << 100 * pc.second << "% " << std::endl;
    std::cout << "Learn step : " << learnStep << std::endl;
    std::cout << std::endl;
    nn.save("neuralNetwork");
    teach(nn, base, learnStep);
    std::random_shuffle(base.begin(), base.end());
  }

  nn.save("neuralNetwork");
}
