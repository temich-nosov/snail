#include "FileTools.h"

void writeInt(std::ostream & stream, int val) {
  stream.write((char *)&val, sizeof(int));
}


void writeFloat(std::ostream & stream, float val) {
  stream.write((char *)&val, sizeof(float));
}


int readInt(std::istream & stream) {
  int res;
  stream.read((char *)&res, sizeof(int));
  return res;
}


float readFloat(std::istream & stream) {
  float res;
  stream.read((char *)&res, sizeof(float));
  return res;
}
