#ifndef ARRAY_BASE_HH
#define ARRAY_BASE_HH

struct ArrayBase {
  inline ArrayBase() {}
  virtual ~ArrayBase() {}
  virtual void erff(float *a, size_t n) noexcept = 0;
  virtual void expf(float *a, size_t n) noexcept = 0;
  virtual void tanhf(float *a, size_t n) noexcept = 0;
};

#endif // ARRAY_BASE_HH
