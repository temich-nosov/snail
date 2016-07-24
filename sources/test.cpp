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
      std::cout << ((data.at(i, j, 0) > 0.3) ? '+' : ' ');
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

    /*
    if (i % 1000 == 0) {
      std::cout << (int)labels[i] << std::endl;
      printDigit(base[i].first);
    }
    */
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

void printOutput(DataArray & d) {
  for (int i = 0; i < d.getSize().d; ++i)
    std::cout << d.at(0, 0, i) << " ";
  std::cout << std::endl;
}

float check(NeuralNetwork & neuralNetwork, std::vector< std::pair<DataArray, int> > & data, bool print) {
  DataArray output(1, 1, 10);
  int cnt = 0;
  for (int i = 0; i < data.size(); ++i) {
    neuralNetwork.propagate(data[i].first, output);
    if (getVal(output) == data[i].second)
      ++cnt;
  }

  return float(cnt) / float(data.size());
}


float check2(NeuralNetwork & neuralNetwork, std::vector< std::pair<DataArray, int> > & data, bool print) {
  DataArray output(1, 1, 1);
  int cnt = 0;
  for (int i = 0; i < data.size(); ++i) {
    neuralNetwork.propagate(data[i].first, output);
    if ((output.at(0, 0, 0) < 0.5) == (data[i].second == 0))
      ++cnt;
  }

  return float(cnt) / float(data.size());
}

void teach(NeuralNetwork & neuralNetwork, std::vector< std::pair<DataArray, int> > & data) {
  DataArray output(1, 1, 10);
  // DataArray output(1, 1, 1);
  // std::random_shuffle(data.begin(), data.end());
  for (int i = 0; i < data.size(); ++i) {
    // std::cout.flush();
    // output.at(0, 0, 0) = (data[i].second ? 1.f : 0.f);
    genOut(data[i].second, output);
    neuralNetwork.backPropagate(data[i].first, output, 0.001);
  }

  // std::cout << std::endl;
}


void teach2(NeuralNetwork & neuralNetwork, std::vector< std::pair<DataArray, int> > & data) {
  DataArray output(1, 1, 1);
  // std::random_shuffle(data.begin(), data.end());
  for (int i = 0; i < data.size(); ++i) {
    output.at(0, 0, 0) = (data[i].second ? 1.f : 0.f);
    neuralNetwork.backPropagate(data[i].first, output, 0.01);
  }
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
  base[3].second = 0;

  NeuralNetwork nn;

  ConvolutionalLayer * l = new ConvolutionalLayer(DataArray::Size(1, 1, 2), 2, 1, 0, 1);
  ConvolutionalLayer * l2 = new ConvolutionalLayer(DataArray::Size(1, 1, 2), 1, 1, 0, 1);

  nn.addLayer(l);
  nn.addLayer(l2);

  for (int i = 0; i < 10000000; ++i) {
    teach(nn, base);
    float pc = check(nn, base, i % 10000 == 0);
    if (i % 10000 == 0) {
      std::cout << i << std::endl;
      std::cout << pc << std::endl;
    }  
  }
}

DataArray genLine(bool vert) {
  DataArray res(5, 5, 1);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      int dist = abs((vert ? i : j) - 2) + 1;
      float d = dist / 6.f;
      res.at(i, j, 0) = d - 1 / 12.f + 2.f / 12.f * rand() / RAND_MAX;
    }
  }

  return res;
}

void simpleTest2() {
  // Научить сеть распознавать палочки вертикальные и горизонтальные
  // Для начала надо создать базу примеров
  // Примеры будут размером 5x5
  std::vector< std::pair<DataArray, int> > base;
  DataArray r;
  for (int i = 0; i < 10; ++i) {
    r = genLine(i % 2);
    base.push_back(std::pair<DataArray, int>(r, i % 2));
  }


  bool ok = true;
  // Создадим сеть
  NeuralNetwork nn;
  nn.addLayer(new ConvolutionalLayer(DataArray::Size(5, 5, 1), 1, 1, 0, 3));
  nn.addLayer(new ConvolutionalLayer(DataArray::Size(3, 3, 1), 1, 3, 0, 3));

  for (int i = 0; i < 300; ++i) {
    float pc = check2(nn, base, i % 10 == 0);
    if (i % 10 == 0) {
      std::cout << i << std::endl;
      std::cout << pc << std::endl;
    }
    teach2(nn, base);
  }
}

DataArray resizeHalf(const DataArray & t) {
  int w = t.getSize().w / 2;
  int h = t.getSize().h / 2;
  DataArray res(t.getSize().w / 2, t.getSize().h / 2, 1);
  for (int i = 0; i < w; ++i) {
    for (int j = 0; j < h; ++j) {
      float c;
      c  = t.at(i * 2, j * 2, 0);
      c += t.at(i * 2 + 1, j * 2, 0);
      c += t.at(i * 2, j * 2 + 1, 0);
      c += t.at(i * 2 + 1, j * 2 + 1, 0);

      res.at(i, j, 0) = c / 4;
    }
  }

  return res;
}

void simpleTest3(const std::vector< std::pair<DataArray, int> > & base) {
  std::vector< std::pair<DataArray, int> > base2;
  std::copy_if(base.begin(), base.end(), std::back_inserter(base2),
      [](const std::pair<DataArray, int> & p) {
        return p.second == 2 || p.second == 3;
      });

  std::cout << base2.size() << std::endl;
  for (int i = 0; i < base2.size(); ++i) {
    base2[i].first = resizeHalf(base2[i].first);
    base2[i].second -= 2;
  }

  NeuralNetwork nn;
  nn.addLayer(new ConvolutionalLayer(DataArray::Size(14, 14, 1), 1, 14, 0, 14));

  for (int i = 0; i < 100000; ++i) {
    float pc = check2(nn, base2, i % 100 == 0);
    if (i % 100 == 0) {
      std::cout << i << std::endl;
      std::cout << pc << std::endl;
    }
    teach2(nn, base2);
  }
}

int main(int argc, char ** argv) {
  if (argc != 3) {
    std::cout << "Необходимо передать имя файла с базой MNIST в качестве параметра" << std::endl;
    return 0;
  }

  std::vector< std::pair<DataArray, int> > base;
  getMNIST(base, argv[1], argv[2]);

  std::random_shuffle(base.begin(), base.end());
  std::vector< std::pair<DataArray, int> > checkBase(100);

  for (int i = 0; i < 100; ++i)
    checkBase[i] = *(base.end() - i - 1);
  base.erase(base.end() - 100, base.end());

  NeuralNetwork nn;
  nn.load("neuralNetwork");
  // nn.addLayer(new ConvolutionalLayer(DataArray::Size(28, 28, 1), 10, 28, 0, 28));
  // nn.addLayer(new FullConnectedLayer(DataArray::Size(28, 28, 1), 10));

  for (int i = 0; i < 2; ++i) {
    float pc = check(nn, checkBase, true);
    std::cout << i  << std::endl;
    std::cout << pc << std::endl;
    teach(nn, base);
  }

  nn.save("neuralNetwork");
}
