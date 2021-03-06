#ifndef ARRAY_BASE_HH
#define ARRAY_BASE_HH

struct ArrayBase {
  inline ArrayBase() {}
  virtual ~ArrayBase() {}
  virtual void erf(double *a, size_t n) noexcept = 0;
  virtual void erff(float *a, size_t n) noexcept = 0;
  virtual void exp(double *a, size_t n) noexcept = 0;
  virtual void expf(float *a, size_t n) noexcept = 0;
  virtual void gelu(double *a, size_t n) noexcept = 0;
  virtual void gelu_backward(double* a, size_t n) noexcept = 0;
  virtual void geluf(float *a, size_t n) noexcept = 0;
  virtual void geluf_backward(float* a, size_t n) noexcept = 0;
  virtual void logistic_cdf(double *a, size_t n) noexcept = 0;
  virtual void logistic_cdff(float *a, size_t n) noexcept = 0;
  virtual void swish(double *a, size_t n) noexcept = 0;
  virtual void swish_backward(double* a, size_t n) noexcept = 0;
  virtual void swishf(float *a, size_t n) noexcept = 0;
  virtual void swishf_backward(float* a, size_t n) noexcept = 0;
  virtual void tanh(double *a, size_t n) noexcept = 0;
  virtual void tanhf(float *a, size_t n) noexcept = 0;
};

#endif // ARRAY_BASE_HH
