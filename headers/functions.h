#pragma once

#include <cmath>

namespace snail {
float func(float x);       /// Сигмоидальная функция
float derivative(float x); /// И её производная

float softPlusFunction(float x); /// Функция активации
float derivativeSoftPlusFunction(float x); /// И её производная
} // namespace snail
