#ifndef LAYER_H
#define LAYER_H

#include "DataArray.h"

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
  virtual void propagate(const DataArray & input, DataArray & output) = 0;

  /// Обратное распространение ошибки
  virtual void backPropagate(const DataArray & input, const DataArray & output, const DataArray & error, DataArray & inputError, float lambda) = 0;

  /// Деструктор
  /**
   * Виртуальный деструктор нужен для возможности удалить
   * по ссылке на базовый класс объект наследованного класса
   */
  virtual ~Layer() {}
};

#endif
