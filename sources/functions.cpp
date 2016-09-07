#include "functions.h"

namespace snail {
float derivative(float x) {
  float f = func(x);
  return f * (1.f - f);
}

float func(float x) {
  return 1.f / (1.f + exp(-x));
}

float softPlusFunction(float x) {
  return log(1.f + exp(x));
}

float derivativeSoftPlusFunction(float x) {
  return 1.f / (1.f + exp(-x));
}
} // namespace snail
