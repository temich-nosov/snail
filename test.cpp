#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include <fstream>

#include "snail.h"

int readInt(std::ifstream & file) {
  char data[4];
  file.read(data, 4);
  std::swap(data[0], data[3]);
  std::swap(data[1], data[2]);
  return *((int *)data);
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

void printDigit(DataArray & data) {
  for (int i = 0; i < data.getSize().w; ++i) {
    for (int j = 0; j < data.getSize().h; ++j) {
      std::cout << ((data.at(i, j, 0) > 0.5) ? '+' : ' ');
    }
    std::cout << std::endl;
  }
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

  std::cout << cnt << std::endl;

  int rows = readInt(baseFile);
  int cols = readInt(baseFile);

  unsigned char * labels = new unsigned char[cnt];
  lableFile.read((char *)labels, cnt);

  base.resize(cnt);
  for (int i = 0; i < cnt; ++i) {
    // std::cout << i + 1 << " of " << cnt << std::endl;
    DataArray data(rows, cols, 1);
    readImage(data, baseFile, rows, cols);
    base[i].first = data;
    base[i].second = labels[i];

    if (i % 1000 == 0) {
      std::cout << (int)labels[i] << std::endl;
      printDigit(base[i].first);
    }
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

float check(NeuralNetwork & neuralNetwork, std::vector< std::pair<DataArray, int> > & data, bool print) {
  // DataArray output(1, 1, 10);
  DataArray output(1, 1, 1);
  int cnt = 0;
  for (int i = 0; i < data.size(); ++i) {
    neuralNetwork.propagate(data[i].first, output);
    if (print)
      std::cout << data[i].first.at(0, 0, 0) << " " << data[i].first.at(0, 0, 1) << " " << output.at(0, 0, 0) << std::endl;
    if ((output.at(0, 0, 0) < 0.5) == (data[i].second == 0))
      ++cnt;
    // if (getVal(output) == data[i].second) {
    //   ++cnt;
      // std::cout << output.at(0, 0, 0) << " " << output.at(0, 0, 1) << std::endl;
    // }
  }

  return float(cnt) / float(data.size());
}


void teach(NeuralNetwork & neuralNetwork, std::vector< std::pair<DataArray, int> > & data) {
  // DataArray output(1, 1, 10);
  DataArray output(1, 1, 1);
  std::random_shuffle(data.begin(), data.end());
  for (int i = 0; i < data.size(); ++i) {
    std::cout.flush();
    output.at(0, 0, 0) = (data[i].second ? 1.f : 0.f);
    // genOut(data[i].second, output);
    neuralNetwork.backPropagate(data[i].first, output, 0.1);
  }

  // std::cout << std::endl;
}

void simpleTest() {
  std::vector< std::pair<DataArray, int> > base;
  base.assign(4, std::pair<DataArray, int>(DataArray(1, 1, 2), 0));
  base[0].first.at(0, 0, 0) = 0;
  base[0].first.at(0, 0, 1) = 0;
  base[0].second = 0;

  base[1].first.at(0, 0, 0) = 0;
  base[1].first.at(0, 0, 1) = 1;
  base[1].second = 1;

  base[2].first.at(0, 0, 0) = 1;
  base[2].first.at(0, 0, 1) = 0;
  base[2].second = 1;

  base[3].first.at(0, 0, 0) = 1;
  base[3].first.at(0, 0, 1) = 1;
  base[3].second = 1;

  NeuralNetwork nn;
  bool ok = true;

  ConvolutionalLayer * l = new ConvolutionalLayer(DataArray::Size(1, 1, 2), 1, 1, 0, 1);
  ok = ok && nn.addLayer(l);
  std::cout << ok << std::endl;
  l->filters[0].first.at(0, 0, 1) = 0.75;
  for (int i = 0; i < 1000000; ++i) {
    teach(nn, base);
    float pc = check(nn, base, i % 10000 == 0);
    if (i % 10000 == 0) {
      std::cout << i << std::endl;
      std::cout << pc << std::endl;
      // std::cout << "Filters size : " << l->filters.size() << std::endl;
      // std::cout << l->filters[0].first.at(0, 0, 0) << " " << l->filters[0].first.at(0, 0, 1) << " " << l->filters[0].second << std::endl;
      // std::cout << l->filters[1].first.at(0, 0, 0) << " " << l->filters[1].second << std::endl;
    }  
  }
}

int main(int argc, char ** argv) {
  simpleTest();
  return 0;

  if (argc != 3) {
    std::cout << "Необходимо передать имя файла с базой MNIST в качестве параметра" << std::endl;
    return 0;
  }

  std::vector< std::pair<DataArray, int> > base;
  getMNIST(base, argv[1], argv[2]);
  // std::random_shuffle(base.begin(), base.end());

  base.resize(50);

  NeuralNetwork nn;
  bool ok = true;
  /*
  ok = ok && nn.addLayer(new ConvolutionalLayer(DataArray::Size(28, 28, 1), 2, 1, 1, 3));
  ok = ok && nn.addLayer(new ReluLayer(DataArray::Size(28, 28, 2)));
  ok = ok && nn.addLayer(new ConvolutionalLayer(DataArray::Size(28, 28, 2), 2, 2, 0, 2));
  // ok = ok && nn.addLayer(new MaxPoolLayer(DataArray::Size(28, 28, 2), 2));
  ok = ok && nn.addLayer(new ConvolutionalLayer(DataArray::Size(14, 14, 2), 4, 1, 0, 3));
  ok = ok && nn.addLayer(new ReluLayer(DataArray::Size(12, 12, 4)));
  ok = ok && nn.addLayer(new ConvolutionalLayer(DataArray::Size(12, 12, 4), 8, 2, 0, 2));
  ok = ok && nn.addLayer(new ReluLayer(DataArray::Size(6, 6, 8)));
  ok = ok && nn.addLayer(new ConvolutionalLayer(DataArray::Size(6, 6, 8), 8, 2, 0, 2));
  ok = ok && nn.addLayer(new ReluLayer(DataArray::Size(3, 3, 8)));
  ok = ok && nn.addLayer(new ConvolutionalLayer(DataArray::Size(3, 3, 8), 10, 3, 0, 3));
  */

  ok = ok && nn.addLayer(new ConvolutionalLayer(DataArray::Size(28, 28, 1), 2, 6, 1, 6));
  ok = ok && nn.addLayer(new ConvolutionalLayer(DataArray::Size(5, 5, 2), 10, 5, 0, 5));
  std::cout << ok << std::endl;
  std::cout << std::string(60, '+') << std::endl;
  for (int i = 0; i < 1000000; ++i) {
    std::cout << i << std::endl;
    teach(nn, base);
    // float pc = check(nn, base);
    // std::cout << pc << std::endl;
  }
}
