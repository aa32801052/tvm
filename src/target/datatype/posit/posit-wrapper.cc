/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file 3rdparty/posit/posit-wrapper.cc
 * \brief Generic wrapper over the Stillwater Universal library for Bring Your Own Datatypes tests
 */
#include <tvm/runtime/base.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "universal/number/posit/posit.hpp"
#include "universal/number/posit/math/error_and_gamma.hpp"
#include "universal/number/posit/math/exponent.hpp"
#include "universal/number/posit/math/hyperbolic.hpp"
#include "universal/number/posit/math/logarithm.hpp"
#include "universal/number/posit/math/minmax.hpp"
#include "universal/number/posit/math/sqrt.hpp"
#include "universal/number/posit/math/trigonometry.hpp"
#include "universal/number/posit/numeric_limits.hpp"
#include "universal/number/posit/quire.hpp"

namespace posit_generic {

template <unsigned bits>
struct storage_for {
  static_assert(bits >= 3 && bits <= 64, "supported generic posit bits are [3, 64]");
  using type = std::conditional_t<
      (bits <= 8), uint8_t,
      std::conditional_t<(bits <= 16), uint16_t,
                         std::conditional_t<(bits <= 32), uint32_t, uint64_t>>>;
};

template <unsigned bits>
using storage_t = typename storage_for<bits>::type;

template <unsigned bits, unsigned es>
using posit_t = sw::universal::posit<bits, es>;

template <typename Storage, unsigned bits>
Storage bitblock_to_storage(const sw::universal::bitblock<bits>& bb) {
  Storage value = 0;
  for (unsigned i = 0; i < bits; ++i) {
    if (bb[i]) {
      value |= (Storage{1} << i);
    }
  }
  return value;
}

template <typename Storage, unsigned bits>
sw::universal::bitblock<bits> storage_to_bitblock(Storage value) {
  sw::universal::bitblock<bits> bb;
  for (unsigned i = 0; i < bits; ++i) {
    bb[i] = static_cast<bool>((value >> i) & Storage{1});
  }
  return bb;
}

template <unsigned bits, unsigned es>
storage_t<bits> bits_of(const posit_t<bits, es>& p) {
  return bitblock_to_storage<storage_t<bits>, bits>(p.get());
}

template <unsigned bits, unsigned es>
posit_t<bits, es> from_bits(storage_t<bits> value) {
  posit_t<bits, es> p;
  p.setBitblock(storage_to_bitblock<storage_t<bits>, bits>(value));
  return p;
}

template <unsigned bits, unsigned es>
storage_t<bits> from_float(float value) {
  return bits_of<bits, es>(posit_t<bits, es>(value));
}

template <unsigned bits, unsigned es>
storage_t<bits> from_double(double value) {
  return bits_of<bits, es>(posit_t<bits, es>(value));
}

template <unsigned bits, unsigned es>
float to_float(storage_t<bits> value) {
  return static_cast<float>(from_bits<bits, es>(value));
}

template <unsigned bits, unsigned es>
double to_double(storage_t<bits> value) {
  return static_cast<double>(from_bits<bits, es>(value));
}

template <unsigned bits, unsigned es>
storage_t<bits> min_value() {
  return bits_of<bits, es>(std::numeric_limits<posit_t<bits, es>>::lowest());
}

template <unsigned bits, unsigned es>
storage_t<bits> add(storage_t<bits> a, storage_t<bits> b) {
  return bits_of<bits, es>(from_bits<bits, es>(a) + from_bits<bits, es>(b));
}

template <unsigned bits, unsigned es>
storage_t<bits> sub(storage_t<bits> a, storage_t<bits> b) {
  return bits_of<bits, es>(from_bits<bits, es>(a) - from_bits<bits, es>(b));
}

template <unsigned bits, unsigned es>
storage_t<bits> mul(storage_t<bits> a, storage_t<bits> b) {
  return bits_of<bits, es>(from_bits<bits, es>(a) * from_bits<bits, es>(b));
}

template <unsigned bits, unsigned es>
storage_t<bits> div(storage_t<bits> a, storage_t<bits> b) {
  return bits_of<bits, es>(from_bits<bits, es>(a) / from_bits<bits, es>(b));
}

template <unsigned bits, unsigned es>
storage_t<bits> fma(storage_t<bits> a, storage_t<bits> b, storage_t<bits> c) {
  auto pa = from_bits<bits, es>(a);
  auto pb = from_bits<bits, es>(b);
  auto pc = from_bits<bits, es>(c);
  return bits_of<bits, es>(pa * pb + pc);
}

template <unsigned bits, unsigned es>
storage_t<bits> max(storage_t<bits> a, storage_t<bits> b) {
  return bits_of<bits, es>(sw::universal::max(from_bits<bits, es>(a), from_bits<bits, es>(b)));
}

template <unsigned bits, unsigned es>
storage_t<bits> min(storage_t<bits> a, storage_t<bits> b) {
  return bits_of<bits, es>(sw::universal::min(from_bits<bits, es>(a), from_bits<bits, es>(b)));
}

template <unsigned bits, unsigned es>
storage_t<bits> sqrt(storage_t<bits> a) {
  return bits_of<bits, es>(sw::universal::sqrt(from_bits<bits, es>(a)));
}

template <unsigned bits, unsigned es>
storage_t<bits> exp(storage_t<bits> a) {
  return bits_of<bits, es>(sw::universal::exp(from_bits<bits, es>(a)));
}

template <unsigned bits, unsigned es>
storage_t<bits> log(storage_t<bits> a) {
  return bits_of<bits, es>(sw::universal::log(from_bits<bits, es>(a)));
}

template <unsigned bits, unsigned es>
storage_t<bits> pow(storage_t<bits> a, storage_t<bits> b) {
  return bits_of<bits, es>(sw::universal::pow(from_bits<bits, es>(a), from_bits<bits, es>(b)));
}

template <unsigned bits, unsigned es>
storage_t<bits> sigmoid(storage_t<bits> a) {
  auto p = from_bits<bits, es>(a);
  auto one = posit_t<bits, es>(1.0f);
  return bits_of<bits, es>(one / (one + sw::universal::exp(-p)));
}

template <unsigned bits, unsigned es>
storage_t<bits> tanh(storage_t<bits> a) {
  return bits_of<bits, es>(sw::universal::tanh(from_bits<bits, es>(a)));
}

template <unsigned bits, unsigned es>
storage_t<bits> cos(storage_t<bits> a) {
  return bits_of<bits, es>(sw::universal::cos(from_bits<bits, es>(a)));
}

template <unsigned bits, unsigned es>
storage_t<bits> sin(storage_t<bits> a) {
  return bits_of<bits, es>(sw::universal::sin(from_bits<bits, es>(a)));
}

template <unsigned bits, unsigned es>
storage_t<bits> tan(storage_t<bits> a) {
  return bits_of<bits, es>(sw::universal::tan(from_bits<bits, es>(a)));
}

template <unsigned bits, unsigned es>
storage_t<bits> erf(storage_t<bits> a) {
  return bits_of<bits, es>(sw::universal::erf(from_bits<bits, es>(a)));
}

template <unsigned bits, unsigned es>
storage_t<bits> softmax(storage_t<bits> a) {
  auto p = from_bits<bits, es>(a);
  auto exp_p = sw::universal::exp(p);
  return bits_of<bits, es>(exp_p / exp_p);
}

template <unsigned bits, unsigned es>
storage_t<bits> bool_to_posit(uint8_t in) {
  return bits_of<bits, es>(posit_t<bits, es>(in != 0));
}

template <unsigned bits, unsigned es>
uint8_t posit_to_bool(storage_t<bits> in) {
  return from_bits<bits, es>(in) == posit_t<bits, es>(0.0f) ? uint8_t{0} : uint8_t{1};
}

template <unsigned bits, unsigned es>
storage_t<bits> int_to_posit(int32_t in) {
  return bits_of<bits, es>(posit_t<bits, es>(in));
}

template <unsigned bits, unsigned es>
int32_t posit_to_int(storage_t<bits> in) {
  return static_cast<int32_t>(from_bits<bits, es>(in));
}

template <unsigned bits, unsigned es>
void quire_matmul(storage_t<bits>* A, int64_t a_offset, int64_t K, storage_t<bits>* B,
                  int64_t b_offset, int64_t col, int64_t N, storage_t<bits>* C,
                  int64_t c_offset) {
  sw::universal::quire<bits, es> q;
  q.clear();

  for (int64_t k = 0; k < K; ++k) {
    auto pa = from_bits<bits, es>(A[a_offset + k]);
    auto pb = from_bits<bits, es>(B[b_offset + k * N + col]);
    q += (pa * pb).to_value();
  }

  posit_t<bits, es> result;
  sw::universal::convert(q.to_value(), result);
  C[c_offset] = bits_of<bits, es>(result);
}

template <unsigned bits, unsigned es>
void quire_matmul_elem(storage_t<bits>* A, int64_t K, storage_t<bits>* B, int64_t col, int64_t N,
                       storage_t<bits>* C, int64_t c_offset) {
  quire_matmul<bits, es>(A, /*a_offset=*/0, K, B, /*b_offset=*/0, col, N, C, c_offset);
}

}  // namespace posit_generic

#define TVM_POSIT_BITS_3_64(V, es) \
  V(3, es) V(4, es) V(5, es) V(6, es) V(7, es) V(8, es) V(9, es) V(10, es) V(11, es) \
  V(12, es) V(13, es) V(14, es) V(15, es) V(16, es) V(17, es) V(18, es) V(19, es) \
  V(20, es) V(21, es) V(22, es) V(23, es) V(24, es) V(25, es) V(26, es) V(27, es) \
  V(28, es) V(29, es) V(30, es) V(31, es) V(32, es) V(33, es) V(34, es) V(35, es) \
  V(36, es) V(37, es) V(38, es) V(39, es) V(40, es) V(41, es) V(42, es) V(43, es) \
  V(44, es) V(45, es) V(46, es) V(47, es) V(48, es) V(49, es) V(50, es) V(51, es) \
  V(52, es) V(53, es) V(54, es) V(55, es) V(56, es) V(57, es) V(58, es) V(59, es) \
  V(60, es) V(61, es) V(62, es) V(63, es) V(64, es)

#define DEFINE_POSIT_GENERIC(bits, E)                                                     \
  TVM_DLL posit_generic::storage_t<bits> BoolToPosit##bits##es##E(uint8_t in) {          \
    return posit_generic::bool_to_posit<bits, E>(in);                                     \
  }                                                                                         \
  TVM_DLL uint8_t Posit##bits##es##E##ToBool(posit_generic::storage_t<bits> in) {         \
    return posit_generic::posit_to_bool<bits, E>(in);                                     \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Uint##bits##ToPosit##bits##es##E(                \
      posit_generic::storage_t<bits> in) {                                                  \
    return in;                                                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##ToUint##bits(                \
      posit_generic::storage_t<bits> in) {                                                  \
    return in;                                                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> IntToPosit##bits##es##E(int32_t in) {            \
    return posit_generic::int_to_posit<bits, E>(in);                                      \
  }                                                                                         \
  TVM_DLL int32_t Posit##bits##es##E##ToInt(posit_generic::storage_t<bits> in) {          \
    return posit_generic::posit_to_int<bits, E>(in);                                      \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> FloatToPosit##bits##es##E(float in) {            \
    return posit_generic::from_float<bits, E>(in);                                        \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> DoubleToPosit##bits##es##E(double in) {          \
    return posit_generic::from_double<bits, E>(in);                                       \
  }                                                                                         \
  TVM_DLL float Posit##bits##es##E##ToFloat(posit_generic::storage_t<bits> in) {          \
    return posit_generic::to_float<bits, E>(in);                                          \
  }                                                                                         \
  TVM_DLL double Posit##bits##es##E##ToDouble(posit_generic::storage_t<bits> in) {        \
    return posit_generic::to_double<bits, E>(in);                                         \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Add(                         \
      posit_generic::storage_t<bits> a, posit_generic::storage_t<bits> b) {               \
    return posit_generic::add<bits, E>(a, b);                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Sub(                         \
      posit_generic::storage_t<bits> a, posit_generic::storage_t<bits> b) {               \
    return posit_generic::sub<bits, E>(a, b);                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Mul(                         \
      posit_generic::storage_t<bits> a, posit_generic::storage_t<bits> b) {               \
    return posit_generic::mul<bits, E>(a, b);                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Div(                         \
      posit_generic::storage_t<bits> a, posit_generic::storage_t<bits> b) {               \
    return posit_generic::div<bits, E>(a, b);                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##FMA(                         \
      posit_generic::storage_t<bits> a, posit_generic::storage_t<bits> b,                 \
      posit_generic::storage_t<bits> c) {                                                 \
    return posit_generic::fma<bits, E>(a, b, c);                                          \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Max(                         \
      posit_generic::storage_t<bits> a, posit_generic::storage_t<bits> b) {               \
    return posit_generic::max<bits, E>(a, b);                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Min(                         \
      posit_generic::storage_t<bits> a, posit_generic::storage_t<bits> b) {               \
    return posit_generic::min<bits, E>(a, b);                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Sqrt(                        \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::sqrt<bits, E>(a);                                               \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Pow(                         \
      posit_generic::storage_t<bits> a, posit_generic::storage_t<bits> b) {               \
    return posit_generic::pow<bits, E>(a, b);                                              \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Exp(                         \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::exp<bits, E>(a);                                                \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Log(                         \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::log<bits, E>(a);                                                \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Sigmoid(                     \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::sigmoid<bits, E>(a);                                            \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Tanh(                        \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::tanh<bits, E>(a);                                               \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Cos(                         \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::cos<bits, E>(a);                                                \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Sin(                         \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::sin<bits, E>(a);                                                \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Tan(                         \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::tan<bits, E>(a);                                                \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Erf(                         \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::erf<bits, E>(a);                                                \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##Softmax(                     \
      posit_generic::storage_t<bits> a) {                                                 \
    return posit_generic::softmax<bits, E>(a);                                            \
  }                                                                                         \
  TVM_DLL posit_generic::storage_t<bits> Posit##bits##es##E##MinValue() {                 \
    return posit_generic::min_value<bits, E>();                                           \
  }                                                                                         \
  TVM_DLL void Posit##bits##es##E##QuireMatmulElem(                                       \
      posit_generic::storage_t<bits>* A, int64_t K,                                       \
      posit_generic::storage_t<bits>* B, int64_t col, int64_t N,                          \
      posit_generic::storage_t<bits>* C, int64_t c_offset) {                              \
    posit_generic::quire_matmul_elem<bits, E>(A, K, B, col, N, C, c_offset);              \
  }                                                                                         \
  TVM_DLL void Posit##bits##es##E##QuireMatmul(                                           \
      posit_generic::storage_t<bits>* A, int64_t a_offset, int64_t K,                     \
      posit_generic::storage_t<bits>* B, int64_t b_offset, int64_t col, int64_t N,        \
      posit_generic::storage_t<bits>* C, int64_t c_offset) {                              \
    posit_generic::quire_matmul<bits, E>(A, a_offset, K, B, b_offset, col, N, C,          \
                                          c_offset);                                        \
  }

extern "C" {

TVM_POSIT_BITS_3_64(DEFINE_POSIT_GENERIC, 0)
TVM_POSIT_BITS_3_64(DEFINE_POSIT_GENERIC, 1)
TVM_POSIT_BITS_3_64(DEFINE_POSIT_GENERIC, 2)
TVM_POSIT_BITS_3_64(DEFINE_POSIT_GENERIC, 3)
TVM_POSIT_BITS_3_64(DEFINE_POSIT_GENERIC, 4)
TVM_POSIT_BITS_3_64(DEFINE_POSIT_GENERIC, 5)

}  // extern "C"

#undef DEFINE_POSIT_GENERIC
#undef TVM_POSIT_BITS_3_64
