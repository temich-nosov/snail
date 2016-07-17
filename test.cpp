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

int main(int argc, char ** argv) {
  if (argc != 3) {
    std::cout << "Необходимо передать имя файла с базой MNIST в качестве параметра" << std::endl;
    return 0;
  }

  std::vector< std::pair<DataArray, int> > base;
  getMNIST(base, argv[1], argv[2]);
}
