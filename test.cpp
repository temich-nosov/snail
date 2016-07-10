#include <iostream>
#include "snail.h"

int main() {
  DataArray x(1, 2, 3);
  DataArray y(1, 2, 3);
  y.clear();
  x = y;
  std::cout << x.at(0, 1, 2) << std::endl;
}
