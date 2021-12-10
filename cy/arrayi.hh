#ifndef ARRAYI_H_
#define ARRAYI_H_

struct ArrayI {
  inline ArrayI() {}
  virtual ~ArrayI() {}
  virtual void erff(float *a, size_t n) noexcept = 0;
  virtual void expf(float *a, size_t n) noexcept = 0;
  virtual void tanhf(float *a, size_t n) noexcept = 0;
};

#endif // ARRAYI_H_
