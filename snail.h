#ifndef SNAIL_H
#define SHAIL_H

#include <vector>
#include <cmath>
#include <algorithm>

float func(float x);       /// Сигмоидальная функция
float derivative(float x); /// И её производная


/**
 * Класс для хранения данных, подаваемых на вход сети,
 * или получающихся после прохождения какого либо её слоя
 */
class DataArray {
public:
  struct Size {
    int w;
    int h;
    int d;

    /// Конструктор по умолчанию
    /**
     * Задаёт нулевой размер для данных
     */
    Size() : w(0), h(0), d(0) {}

    /// Конструктор по заданной высоте, ширине и глубине
    /**
     * @param w - Высота (Значение x всегда меньше высоты)
     * @param h - Ширина (Значение y меньше ширины)
     * @param d - Глубина (Значение z меньше глубины)
     */
    Size(int w, int h, int d) : w(w), h(h), d(d) {}

    /// Проверка принадлежности точки массиву заданного размера
    bool check(int x, int y, int z) const;

    bool operator==(const Size & size) const;
    bool operator!=(const Size & size) const;

    /// Кол-во элементов в массиве такого размера 
    int elemCnt() const {
      return w * h * d;
    }
  };

private:
  /// Массив с данными
  float * arr;
  Size size;

public:
  /// Конструктор по умолчанию
  /**
   * Создаст пустой массив данных
   */
  DataArray();

  /// Конструктор массива заданного размера
  DataArray(Size size);

  /// Конструктор массива заданного размера
  DataArray(int w, int h, int d);

  /// Конструктор копирования
  DataArray(const DataArray & dataArray);

  /// Оператор копирования
  DataArray & operator=(const DataArray & other);

  /// Вернёт значение по индексу x, y, z
  float & at(int x, int y, int z);

  /// Вернёт значение по индексу x, y, z
  const float & at(int x, int y, int z) const;

  /// Обнулить массив
  void clear();

  /// Заполнить массив случайными значениями
  void fillRnd(float minVal = 0, float maxVal = 1);

  Size getSize() const {
    return size;
  }

  /// Дополнить рамкой нулей ширины width
  void addZeros(int width, DataArray & output) const;

  void removeFrame(int width, DataArray & output) const;

  /// Деструктор
  ~DataArray();
};

/**
 * Абстрактный класс для слоя
 * От этого класса наследованны все остальные слои
 */
class Layer {
public:
  /// Размерность входных данных для слоя
  virtual DataArray::Size getInputSize() const = 0;

  /// Размерность выходных данных для слоя
  virtual DataArray::Size getOutputSize() const = 0;

  /// Прямое распространение
  virtual void propagate(const DataArray & input, DataArray & output) const = 0;

  /// Обратное распространение ошибки
  virtual void backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda) = 0;

  /// Деструктор
  /**
   * Виртуальный деструктор нужен для возможности удалить
   * по ссылке на базовый класс объект наследованного класса
   */
  virtual ~Layer() = 0;
};

/**
 * Класс свёрточного слоя
 */
class ConvolutionalLayer : public Layer {
  DataArray::Size inputSize;
  DataArray::Size outputSize;

  int depth;       /// Число фильтров в слое
  int stride;      /// Шаг
  int zeroPadding; /// Дополнение нулями
  int filterSize;  /// Размер части, выбираемой фильтром

  std::vector< std::pair<DataArray, float> > filters; // массив весов связей и смещение

public:
  ConvolutionalLayer(DataArray::Size inputSize, int depth, int stride, int zeroPadding, int filterSize);
  
  DataArray::Size getInputSize() const {
    return inputSize;
  }

  DataArray::Size getOutputSize() const {
    return outputSize;
  }

  virtual void propagate(const DataArray & input, DataArray & output) const;
  virtual void backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda);

  ~ConvolutionalLayer() {};
};

/**
 * Класс слоя объединения
 */
class MaxPoolLayer : public Layer {
  DataArray::Size inputSize;
  DataArray::Size outputSize;
  int filterSize;

public:
  DataArray::Size getInputSize() const;
  DataArray::Size getOutputSize() const;

  MaxPoolLayer(DataArray::Size inputSize, int filterSize) : inputSize(inputSize), filterSize(filterSize) {
    outputSize = inputSize;
    outputSize.w /= filterSize;
    outputSize.h /= filterSize;
  }

  virtual void propagate(const DataArray & input, DataArray & output) const;
  virtual void backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda);

  ~MaxPoolLayer();
};

/**
 * Полносвязный слой
 */
class FullConnectedLayer : public ConvolutionalLayer {
public:
  FullConnectedLayer();
  ~FullConnectedLayer();
};

class NeuralNetwork {
  std::vector<Layer *> layers;
  std::vector<DataArray> data;
public:
  NeuralNetwork();
  
  /// Очистить сеть
  void clear();

  /// Загрузить сеть из файла
  bool load(std::string filename);

  /// Сохранить сеть в файл
  bool save(std::string filename);

  /// Добавить слой в сеть
  bool addLayer(Layer * layer);

  /// Пропустить данные через сеть
  void propagate(const DataArray & input, DataArray & output) const;

  /// Произвести итерацию обучения сети
  void backPropagate(const DataArray & input, const DataArray & expectOutput, float lambda);
};

#endif /* SHAIL_H */
