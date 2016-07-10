#ifndef SNAIL_H
#define SHAIL_H

#include <vector>

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
  /**
   * Копирование произойдёт только в том случае,
   * если размеры массивов совпадают.
   * TODO : Выбросить исключение в противном случае
   */
  DataArray & operator=(const DataArray & other);

  /// Вернёт значение по индексу x, y, z
  float & at(int x, int y, int z);

  /// Вернёт значение по индексу x, y, z
  const float & at(int x, int y, int z) const;

  /// Обнулить массив
  void clear();

  /// Деструктор
  ~DataArray();

  Size getSize() const {
    return size;
  }
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
  virtual void backPropagate(const DataArray & input, DataArray & output, float lambda) = 0;

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

  vector<DataArray> filters;

public:
  ConvolutionalLayer(DataArray::Size inputSize, ) {}
  
  DataArray::Size getInputSize() const {
    return inputSize;
  }

  DataArray::Size getOutputSize() const {
    return outputSize;
  }

  virtual void propagate(const DataArray & input, DataArray & output) const;
  virtual void backPropagate(const DataArray & input, DataArray & output, float lambda);

  ~ConvolutionalLayer();
};

/**
 * Класс слоя объединения
 */
class PoolLayer : public Layer {
public:
  DataArray::Size getInputSize() const;
  DataArray::Size getOutputSize() const;

  virtual void propagate(const DataArray & input, DataArray & output) const;
  virtual void backPropagate(const DataArray & input, DataArray & output, float lambda);

  ~PoolLayer();
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
