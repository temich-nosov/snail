#ifndef DATAARRAY_H
#define DATAARRAY_H

#include "FileTools.h"

#include <algorithm>
#include <iostream>

#include <stdexcept>

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

    void write(std::ostream & stream) const;
    static DataArray::Size read(std::istream & stream);
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

  friend void swap( DataArray& a, DataArray& b );

  void write(std::ostream & stream) const;
  static DataArray read(std::istream & stream);
};

void swap( DataArray& a, DataArray& b );

#endif
