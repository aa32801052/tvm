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
 * \brief Wrapper over the Stillwater Universal library for Bring Your Own Datatypes tests
 *
 * To compile TVM with this file,
 * 1. clone the Stillwater Universal repo from here `https://github.com/stillwater-sc/universal`.
 * 2. set `SET_BYODT_POSIT` ON and `UNIVERSAL_PATH` as the path to the folder containing Stillwater
 * Universal in your cmake file
 *
 * TODO(@gussmith23 @hypercubestart) Link to BYODT docs when they exist?
 */
#include <tvm/runtime/base.h>

#include <cstdint>
#include <cstring>
#include <cmath>

#include "universal/number/posit/posit.hpp"
#include "universal/number/posit/math/exponent.hpp"
#include "universal/number/posit/math/hyperbolic.hpp"
#include "universal/number/posit/math/logarithm.hpp"
#include "universal/number/posit/math/sqrt.hpp"
#include "universal/number/posit/numeric_limits.hpp"
#include "universal/number/posit/math/minmax.hpp"
#include "universal/number/posit/math/trigonometry.hpp"
#include "universal/number/posit/math/error_and_gamma.hpp"
#include "universal/number/posit/quire.hpp"

#include <iostream>

extern "C" {

static inline uint16_t posit16_bits(sw::universal::posit<16, 2> p) {
  return static_cast<uint16_t>(p.get().to_ulong());
}
static inline sw::universal::posit<16, 2> posit16_from_bits(uint16_t b) {
  sw::universal::posit<16, 2> p;
  p.setbits(b);
  return p;
}

static inline uint32_t posit32_bits(sw::universal::posit<32, 2> p) {
  return static_cast<uint32_t>(p.get().to_ulong());
}
static inline sw::universal::posit<32, 2> posit32_from_bits(uint32_t b) {
  sw::universal::posit<32, 2> p;
  p.setbits(b);
  return p;
}

static inline uint8_t posit8_bits(const sw::universal::posit<8, 2>& p) {
  return static_cast<uint8_t>(p.get().to_ulong());
}

static inline sw::universal::posit<8, 2> posit8_from_bits(uint8_t b) {
  sw::universal::posit<8, 2> p;
  p.setbits(b);
  return p;
}

TVM_DLL uint16_t BoolToPosit16es2(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<16, 2> p = b;
  return posit16_bits(p);
}

TVM_DLL uint32_t BoolToPosit32es2(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<32, 2> p = b;
  return posit32_bits(p);
}

TVM_DLL uint8_t Posit16es2ToBool(uint16_t in_bits) {
  auto p = posit16_from_bits(in_bits);
  return (p == sw::universal::posit<16, 2>(0.0f)) ? uint8_t{0} : uint8_t{1};
}

TVM_DLL uint8_t Posit32es2ToBool(uint32_t in_bits) {
  auto p = posit32_from_bits(in_bits);
  return (p == sw::universal::posit<32, 2>(0.0f)) ? uint8_t{0} : uint8_t{1};
}

TVM_DLL uint16_t IntToPosit16es2(int32_t in) {
  sw::universal::posit<16, 2> p = in;
  return posit16_bits(p);
}

TVM_DLL uint32_t IntToPosit32es2(int32_t in) {
  sw::universal::posit<32, 2> p = in;
  return posit32_bits(p);
}

TVM_DLL uint8_t IntToPosit8es2(int32_t in) {
  sw::universal::posit<8, 2> p = in;
  return static_cast<uint8_t>(p.get().to_ulong());
}

TVM_DLL int32_t Posit32es2ToInt(uint32_t in_bits) {
  auto p = posit32_from_bits(in_bits);
  return static_cast<int32_t>(p);
}

TVM_DLL int32_t Posit16es2ToInt(uint16_t in_bits) {
  auto p = posit16_from_bits(in_bits);
  return static_cast<int32_t>(p);
}

TVM_DLL int32_t Posit8es2ToInt(uint8_t in_bits) {
  sw::universal::posit<8, 2> p;
  p.setbits(in_bits);
  return static_cast<int32_t>(p);
}

// Identity casts (uint <-> posit with same bitwidth): just return the bits unchanged
TVM_DLL uint8_t Uint8ToPosit8es2(uint8_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint16_t Uint16ToPosit16es2(uint16_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint32_t Uint32ToPosit32es2(uint32_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint8_t Posit8es2ToUint8(uint8_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint16_t Posit16es2ToUint16(uint16_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint32_t Posit32es2ToUint32(uint32_t in) {
  return in;  // No conversion needed, same bit representation
}

// ----- posit16 es2: uint16_t API -----
TVM_DLL uint16_t FloatToPosit16es2(float in) {
  return posit16_bits(sw::universal::posit<16, 2>(in));
}
TVM_DLL float Posit16es2ToFloat(uint16_t in) {
  return static_cast<float>(posit16_from_bits(in));
}
TVM_DLL uint16_t Posit16es2Add(uint16_t a, uint16_t b) {
  return posit16_bits(posit16_from_bits(a) + posit16_from_bits(b));
}
TVM_DLL uint16_t Posit16es2Sub(uint16_t a, uint16_t b) {
  return posit16_bits(posit16_from_bits(a) - posit16_from_bits(b));
}
TVM_DLL uint16_t Posit16es2Mul(uint16_t a, uint16_t b) {
  return posit16_bits(posit16_from_bits(a) * posit16_from_bits(b));
}
TVM_DLL uint16_t Posit16es2Div(uint16_t a, uint16_t b) {
  return posit16_bits(posit16_from_bits(a) / posit16_from_bits(b));
}
TVM_DLL uint16_t Posit16es2FMA(uint16_t a, uint16_t b, uint16_t c) {
  // Use standard posit mul+add which is more accurate than the broken quire FMA
  // The issue with quire FMA is that adding a posit to quire introduces conversion errors
  auto pa = posit16_from_bits(a);
  auto pb = posit16_from_bits(b);
  auto pc = posit16_from_bits(c);
  return posit16_bits(pa * pb + pc);
}
TVM_DLL uint16_t Posit16es2Max(uint16_t a, uint16_t b) {
  auto pa = posit16_from_bits(a), pb = posit16_from_bits(b);
  return posit16_bits(sw::universal::max(pa, pb));
}
TVM_DLL uint16_t Posit16es2Min(uint16_t a, uint16_t b) {
  auto pa = posit16_from_bits(a), pb = posit16_from_bits(b);
  return posit16_bits(sw::universal::min(pa, pb));
}
TVM_DLL uint16_t Posit16es2Sqrt(uint16_t a) {
  return posit16_bits(sw::universal::sqrt(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Exp(uint16_t a) {
  return posit16_bits(sw::universal::exp(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Log(uint16_t a) {
  return posit16_bits(sw::universal::log(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Pow(uint16_t a, uint16_t b) {
  return posit16_bits(sw::universal::pow(posit16_from_bits(a), posit16_from_bits(b)));
}
TVM_DLL uint16_t Posit16es2Sigmoid(uint16_t a) {
  auto p = posit16_from_bits(a);
  return posit16_bits(sw::universal::posit<16, 2>(1.0) / (sw::universal::posit<16, 2>(1.0) + sw::universal::exp(-p)));
}
TVM_DLL uint16_t Posit16es2Tanh(uint16_t a) {
  return posit16_bits(sw::universal::tanh(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Cos(uint16_t a) {
  return posit16_bits(sw::universal::cos(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Sin(uint16_t a) {
  return posit16_bits(sw::universal::sin(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Tan(uint16_t a) {
  return posit16_bits(sw::universal::tan(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Erf(uint16_t a) {
  return posit16_bits(sw::universal::erf(posit16_from_bits(a)));
}
// Cast: posit32(bits)->posit16(bits) and reverse
TVM_DLL uint16_t Posit32ToPosit16es2(uint32_t in_bits) {
  auto p32 = posit32_from_bits(in_bits);
  sw::universal::posit<16, 2> p16 = p32;
  return posit16_bits(p16);
}
TVM_DLL uint32_t Posit16ToPosit32es2(uint16_t in_bits) {
  auto p16 = posit16_from_bits(in_bits);
  sw::universal::posit<32, 2> p32 = p16;
  return posit32_bits(p32);
}

// Cast: posit32/16 <-> posit8 for es2
TVM_DLL uint8_t Posit32ToPosit8es2(uint32_t in_bits) {
  auto p32 = posit32_from_bits(in_bits);
  sw::universal::posit<8, 2> p8 = p32;
  return posit8_bits(p8);
}
TVM_DLL uint8_t Posit16ToPosit8es2(uint16_t in_bits) {
  auto p16 = posit16_from_bits(in_bits);
  sw::universal::posit<8, 2> p8 = p16;
  return posit8_bits(p8);
}
TVM_DLL uint32_t Posit8ToPosit32es2(uint8_t in_bits) {
  auto p8 = posit8_from_bits(in_bits);
  sw::universal::posit<32, 2> p32 = p8;
  return posit32_bits(p32);
}
TVM_DLL uint16_t Posit8ToPosit16es2(uint8_t in_bits) {
  auto p8 = posit8_from_bits(in_bits);
  sw::universal::posit<16, 2> p16 = p8;
  return posit16_bits(p16);
}

// ----- posit32 es2: uint32_t API -----
TVM_DLL uint32_t FloatToPosit32es2(float in) {
  return posit32_bits(sw::universal::posit<32, 2>(in));
}
TVM_DLL float Posit32es2ToFloat(uint32_t in) {
  return static_cast<float>(posit32_from_bits(in));
}
TVM_DLL uint32_t Posit32es2Add(uint32_t a, uint32_t b) {
  return posit32_bits(posit32_from_bits(a) + posit32_from_bits(b));
}
TVM_DLL uint32_t Posit32es2Sub(uint32_t a, uint32_t b) {
  return posit32_bits(posit32_from_bits(a) - posit32_from_bits(b));
}
TVM_DLL uint32_t Posit32es2Mul(uint32_t a, uint32_t b) {
  return posit32_bits(posit32_from_bits(a) * posit32_from_bits(b));
}
TVM_DLL uint32_t Posit32es2Div(uint32_t a, uint32_t b) {
  return posit32_bits(posit32_from_bits(a) / posit32_from_bits(b));
}
TVM_DLL uint32_t Posit32es2FMA(uint32_t a, uint32_t b, uint32_t c) {
  auto pa = posit32_from_bits(a);
  auto pb = posit32_from_bits(b);
  auto pc = posit32_from_bits(c);
  // Use native posit multiply-add without quire to match baseline behavior
  return posit32_bits(pa * pb + pc);
}
TVM_DLL uint32_t Posit32es2Max(uint32_t a, uint32_t b) {
  auto pa = posit32_from_bits(a), pb = posit32_from_bits(b);
  return posit32_bits(sw::universal::max(pa, pb));
}
TVM_DLL uint32_t Posit32es2Min(uint32_t a, uint32_t b) {
  auto pa = posit32_from_bits(a), pb = posit32_from_bits(b);
  return posit32_bits(sw::universal::min(pa, pb));
}
TVM_DLL uint32_t Posit32es2Sqrt(uint32_t a) {
  return posit32_bits(sw::universal::sqrt(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Exp(uint32_t a) {
  return posit32_bits(sw::universal::exp(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Log(uint32_t a) {
  return posit32_bits(sw::universal::log(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Pow(uint32_t a, uint32_t b) {
  return posit32_bits(sw::universal::pow(posit32_from_bits(a), posit32_from_bits(b)));
}
TVM_DLL uint32_t Posit32es2Sigmoid(uint32_t a) {
  auto p = posit32_from_bits(a);
  return posit32_bits(sw::universal::posit<32, 2>(1.0) / (sw::universal::posit<32, 2>(1.0) + sw::universal::exp(-p)));
}
TVM_DLL uint32_t Posit32es2Tanh(uint32_t a) {
  return posit32_bits(sw::universal::tanh(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Cos(uint32_t a) {
  return posit32_bits(sw::universal::cos(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Sin(uint32_t a) {
  return posit32_bits(sw::universal::sin(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Tan(uint32_t a) {
  return posit32_bits(sw::universal::tan(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Erf(uint32_t a) {
  return posit32_bits(sw::universal::erf(posit32_from_bits(a)));
}

// ----- posit8 es2: uint8_t API -----
TVM_DLL uint8_t FloatToPosit8es2(float in) {
  sw::universal::posit<8, 2> p(in);
  return posit8_bits(p);
}
TVM_DLL float Posit8es2ToFloat(uint8_t in) {
  return static_cast<float>(posit8_from_bits(in));
}
TVM_DLL uint8_t Posit8es2Add(uint8_t a, uint8_t b) {
  return posit8_bits(posit8_from_bits(a) + posit8_from_bits(b));
}
TVM_DLL uint8_t Posit8es2Sub(uint8_t a, uint8_t b) {
  return posit8_bits(posit8_from_bits(a) - posit8_from_bits(b));
}
TVM_DLL uint8_t Posit8es2Mul(uint8_t a, uint8_t b) {
  return posit8_bits(posit8_from_bits(a) * posit8_from_bits(b));
}
TVM_DLL uint8_t Posit8es2Div(uint8_t a, uint8_t b) {
  return posit8_bits(posit8_from_bits(a) / posit8_from_bits(b));
}
TVM_DLL uint8_t Posit8es2FMA(uint8_t a, uint8_t b, uint8_t c) {
  auto pa = posit8_from_bits(a);
  auto pb = posit8_from_bits(b);
  auto pc = posit8_from_bits(c);
  return posit8_bits(pa * pb + pc);
}
TVM_DLL uint8_t Posit8es2Max(uint8_t a, uint8_t b) {
  auto pa = posit8_from_bits(a), pb = posit8_from_bits(b);
  return posit8_bits(sw::universal::max(pa, pb));
}
TVM_DLL uint8_t Posit8es2Min(uint8_t a, uint8_t b) {
  auto pa = posit8_from_bits(a), pb = posit8_from_bits(b);
  return posit8_bits(sw::universal::min(pa, pb));
}
TVM_DLL uint8_t Posit8es2Sqrt(uint8_t a) {
  return posit8_bits(sw::universal::sqrt(posit8_from_bits(a)));
}
TVM_DLL uint8_t Posit8es2Exp(uint8_t a) {
  return posit8_bits(sw::universal::exp(posit8_from_bits(a)));
}
TVM_DLL uint8_t Posit8es2Log(uint8_t a) {
  return posit8_bits(sw::universal::log(posit8_from_bits(a)));
}
TVM_DLL uint8_t Posit8es2Pow(uint8_t a, uint8_t b) {
  return posit8_bits(sw::universal::pow(posit8_from_bits(a), posit8_from_bits(b)));
}
TVM_DLL uint8_t Posit8es2Sigmoid(uint8_t a) {
  auto p = posit8_from_bits(a);
  return posit8_bits(sw::universal::posit<8, 2>(1.0) / (sw::universal::posit<8, 2>(1.0) + sw::universal::exp(-p)));
}
TVM_DLL uint8_t Posit8es2Tanh(uint8_t a) {
  return posit8_bits(sw::universal::tanh(posit8_from_bits(a)));
}
TVM_DLL uint8_t Posit8es2Cos(uint8_t a) {
  return posit8_bits(sw::universal::cos(posit8_from_bits(a)));
}
TVM_DLL uint8_t Posit8es2Sin(uint8_t a) {
  return posit8_bits(sw::universal::sin(posit8_from_bits(a)));
}
TVM_DLL uint8_t Posit8es2Tan(uint8_t a) {
  return posit8_bits(sw::universal::tan(posit8_from_bits(a)));
}
TVM_DLL uint8_t Posit8es2Erf(uint8_t a) {
  return posit8_bits(sw::universal::erf(posit8_from_bits(a)));
}
TVM_DLL uint8_t BoolToPosit8es2(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<8, 2> p = b;
  return posit8_bits(p);
}
TVM_DLL uint8_t Posit8es2ToBool(uint8_t in_bits) {
  auto p = posit8_from_bits(in_bits);
  return (p == sw::universal::posit<8, 2>(0.0f)) ? uint8_t{0} : uint8_t{1};
}

// ---- posit es1 support ----
static inline uint16_t posit16es1_bits(sw::universal::posit<16, 1> p) {
  return static_cast<uint16_t>(p.get().to_ulong());
}
static inline sw::universal::posit<16, 1> posit16es1_from_bits(uint16_t b) {
  sw::universal::posit<16, 1> p;
  p.setbits(b);
  return p;
}

static inline uint32_t posit32es1_bits(sw::universal::posit<32, 1> p) {
  return static_cast<uint32_t>(p.get().to_ulong());
}
static inline sw::universal::posit<32, 1> posit32es1_from_bits(uint32_t b) {
  sw::universal::posit<32, 1> p;
  p.setbits(b);
  return p;
}

static inline uint8_t posit8e1_bits(const sw::universal::posit<8, 1>& p) {
  return static_cast<uint8_t>(p.get().to_ulong());
}

static inline sw::universal::posit<8, 1> posit8e1_from_bits(uint8_t b) {
  sw::universal::posit<8, 1> p;
  p.setbits(b);
  return p;
}

// Bool conversions
TVM_DLL uint16_t BoolToPosit16es1(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<16, 1> p = b;
  return posit16es1_bits(p);
}

TVM_DLL uint32_t BoolToPosit32es1(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<32, 1> p = b;
  return posit32es1_bits(p);
}

TVM_DLL uint8_t BoolToPosit8es1(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<8, 1> p = b;
  return posit8e1_bits(p);
}

TVM_DLL uint8_t Posit16es1ToBool(uint16_t in_bits) {
  auto p = posit16es1_from_bits(in_bits);
  return (p == sw::universal::posit<16, 1>(0.0f)) ? uint8_t{0} : uint8_t{1};
}

TVM_DLL uint8_t Posit32es1ToBool(uint32_t in_bits) {
  auto p = posit32es1_from_bits(in_bits);
  return (p == sw::universal::posit<32, 1>(0.0f)) ? uint8_t{0} : uint8_t{1};
}

TVM_DLL uint8_t Posit8es1ToBool(uint8_t in_bits) {
  auto p = posit8e1_from_bits(in_bits);
  return (p == sw::universal::posit<8, 1>(0.0f)) ? uint8_t{0} : uint8_t{1};
}

// Int conversions
TVM_DLL uint16_t IntToPosit16es1(int32_t in) {
  sw::universal::posit<16, 1> p = in;
  return posit16es1_bits(p);
}

TVM_DLL uint32_t IntToPosit32es1(int32_t in) {
  sw::universal::posit<32, 1> p = in;
  return posit32es1_bits(p);
}

TVM_DLL uint8_t IntToPosit8es1(int32_t in) {
  sw::universal::posit<8, 1> p = in;
  return posit8e1_bits(p);
}

TVM_DLL int32_t Posit32es1ToInt(uint32_t in_bits) {
  auto p = posit32es1_from_bits(in_bits);
  return static_cast<int32_t>(p);
}

TVM_DLL int32_t Posit16es1ToInt(uint16_t in_bits) {
  auto p = posit16es1_from_bits(in_bits);
  return static_cast<int32_t>(p);
}

TVM_DLL int32_t Posit8es1ToInt(uint8_t in_bits) {
  auto p = posit8e1_from_bits(in_bits);
  return static_cast<int32_t>(p);
}

// Identity casts (uint <-> posit with same bitwidth)
TVM_DLL uint8_t Uint8ToPosit8es1(uint8_t in) { return in; }
TVM_DLL uint16_t Uint16ToPosit16es1(uint16_t in) { return in; }
TVM_DLL uint32_t Uint32ToPosit32es1(uint32_t in) { return in; }
TVM_DLL uint8_t Posit8es1ToUint8(uint8_t in) { return in; }
TVM_DLL uint16_t Posit16es1ToUint16(uint16_t in) { return in; }
TVM_DLL uint32_t Posit32es1ToUint32(uint32_t in) { return in; }

// ----- posit16 es1: uint16_t API -----
TVM_DLL uint16_t FloatToPosit16es1(float in) { return posit16es1_bits(sw::universal::posit<16, 1>(in)); }
TVM_DLL float Posit16es1ToFloat(uint16_t in) { return static_cast<float>(posit16es1_from_bits(in)); }
TVM_DLL uint16_t Posit16es1Add(uint16_t a, uint16_t b) { return posit16es1_bits(posit16es1_from_bits(a) + posit16es1_from_bits(b)); }
TVM_DLL uint16_t Posit16es1Sub(uint16_t a, uint16_t b) { return posit16es1_bits(posit16es1_from_bits(a) - posit16es1_from_bits(b)); }
TVM_DLL uint16_t Posit16es1Mul(uint16_t a, uint16_t b) { return posit16es1_bits(posit16es1_from_bits(a) * posit16es1_from_bits(b)); }
TVM_DLL uint16_t Posit16es1Div(uint16_t a, uint16_t b) { return posit16es1_bits(posit16es1_from_bits(a) / posit16es1_from_bits(b)); }
TVM_DLL uint16_t Posit16es1Max(uint16_t a, uint16_t b) {
  auto pa = posit16es1_from_bits(a), pb = posit16es1_from_bits(b);
  return posit16es1_bits(sw::universal::max(pa, pb));
}
TVM_DLL uint16_t Posit16es1Min(uint16_t a, uint16_t b) {
  auto pa = posit16es1_from_bits(a), pb = posit16es1_from_bits(b);
  return posit16es1_bits(sw::universal::min(pa, pb));
}
TVM_DLL uint16_t Posit16es1Sqrt(uint16_t a) { return posit16es1_bits(sw::universal::sqrt(posit16es1_from_bits(a))); }
TVM_DLL uint16_t Posit16es1Exp(uint16_t a) { return posit16es1_bits(sw::universal::exp(posit16es1_from_bits(a))); }
TVM_DLL uint16_t Posit16es1Log(uint16_t a) { return posit16es1_bits(sw::universal::log(posit16es1_from_bits(a))); }
TVM_DLL uint16_t Posit16es1Pow(uint16_t a, uint16_t b) {
  return posit16es1_bits(sw::universal::pow(posit16es1_from_bits(a), posit16es1_from_bits(b)));
}
TVM_DLL uint16_t Posit16es1Sigmoid(uint16_t a) {
  auto p = posit16es1_from_bits(a);
  return posit16es1_bits(sw::universal::posit<16, 1>(1.0) / (sw::universal::posit<16, 1>(1.0) + sw::universal::exp(-p)));
}
TVM_DLL uint16_t Posit16es1Tanh(uint16_t a) { return posit16es1_bits(sw::universal::tanh(posit16es1_from_bits(a))); }
TVM_DLL uint16_t Posit16es1Cos(uint16_t a) { return posit16es1_bits(sw::universal::cos(posit16es1_from_bits(a))); }
TVM_DLL uint16_t Posit16es1Sin(uint16_t a) { return posit16es1_bits(sw::universal::sin(posit16es1_from_bits(a))); }
TVM_DLL uint16_t Posit16es1Tan(uint16_t a) { return posit16es1_bits(sw::universal::tan(posit16es1_from_bits(a))); }
TVM_DLL uint16_t Posit16es1Erf(uint16_t a) { return posit16es1_bits(sw::universal::erf(posit16es1_from_bits(a))); }
TVM_DLL uint16_t Posit16es1Softmax(uint16_t a) {
  auto p = posit16es1_from_bits(a);
  auto exp_p = sw::universal::exp(p);
  return posit16es1_bits(exp_p / exp_p);
}
TVM_DLL uint16_t Posit32ToPosit16es1(uint32_t in_bits) {
  auto p32 = posit32es1_from_bits(in_bits);
  sw::universal::posit<16, 1> p16 = p32;
  return posit16es1_bits(p16);
}
TVM_DLL uint32_t Posit16ToPosit32es1(uint16_t in_bits) {
  auto p16 = posit16es1_from_bits(in_bits);
  sw::universal::posit<32, 1> p32 = p16;
  return posit32es1_bits(p32);
}

// Cast: posit32/16 <-> posit8 for es1
TVM_DLL uint8_t Posit32ToPosit8es1(uint32_t in_bits) {
  auto p32 = posit32es1_from_bits(in_bits);
  sw::universal::posit<8, 1> p8 = p32;
  return posit8e1_bits(p8);
}
TVM_DLL uint8_t Posit16ToPosit8es1(uint16_t in_bits) {
  auto p16 = posit16es1_from_bits(in_bits);
  sw::universal::posit<8, 1> p8 = p16;
  return posit8e1_bits(p8);
}
TVM_DLL uint32_t Posit8ToPosit32es1(uint8_t in_bits) {
  auto p8 = posit8e1_from_bits(in_bits);
  sw::universal::posit<32, 1> p32 = p8;
  return posit32es1_bits(p32);
}
TVM_DLL uint16_t Posit8ToPosit16es1(uint8_t in_bits) {
  auto p8 = posit8e1_from_bits(in_bits);
  sw::universal::posit<16, 1> p16 = p8;
  return posit16es1_bits(p16);
}

// ----- posit32 es1: uint32_t API -----
TVM_DLL uint32_t FloatToPosit32es1(float in) { return posit32es1_bits(sw::universal::posit<32, 1>(in)); }
TVM_DLL float Posit32es1ToFloat(uint32_t in) { return static_cast<float>(posit32es1_from_bits(in)); }
TVM_DLL uint32_t Posit32es1Add(uint32_t a, uint32_t b) { return posit32es1_bits(posit32es1_from_bits(a) + posit32es1_from_bits(b)); }
TVM_DLL uint32_t Posit32es1Sub(uint32_t a, uint32_t b) { return posit32es1_bits(posit32es1_from_bits(a) - posit32es1_from_bits(b)); }
TVM_DLL uint32_t Posit32es1Mul(uint32_t a, uint32_t b) { return posit32es1_bits(posit32es1_from_bits(a) * posit32es1_from_bits(b)); }
TVM_DLL uint32_t Posit32es1Div(uint32_t a, uint32_t b) { return posit32es1_bits(posit32es1_from_bits(a) / posit32es1_from_bits(b)); }

TVM_DLL uint32_t Posit32es1Max(uint32_t a, uint32_t b) {
  auto pa = posit32es1_from_bits(a), pb = posit32es1_from_bits(b);
  return posit32es1_bits(sw::universal::max(pa, pb));
}
TVM_DLL uint32_t Posit32es1Min(uint32_t a, uint32_t b) {
  auto pa = posit32es1_from_bits(a), pb = posit32es1_from_bits(b);
  return posit32es1_bits(sw::universal::min(pa, pb));
}
TVM_DLL uint32_t Posit32es1Sqrt(uint32_t a) { return posit32es1_bits(sw::universal::sqrt(posit32es1_from_bits(a))); }
TVM_DLL uint32_t Posit32es1Exp(uint32_t a) { return posit32es1_bits(sw::universal::exp(posit32es1_from_bits(a))); }
TVM_DLL uint32_t Posit32es1Log(uint32_t a) { return posit32es1_bits(sw::universal::log(posit32es1_from_bits(a))); }
TVM_DLL uint32_t Posit32es1Pow(uint32_t a, uint32_t b) {
  return posit32es1_bits(sw::universal::pow(posit32es1_from_bits(a), posit32es1_from_bits(b)));
}
TVM_DLL uint32_t Posit32es1Sigmoid(uint32_t a) {
  auto p = posit32es1_from_bits(a);
  return posit32es1_bits(sw::universal::posit<32, 1>(1.0) / (sw::universal::posit<32, 1>(1.0) + sw::universal::exp(-p)));
}
TVM_DLL uint32_t Posit32es1Tanh(uint32_t a) { return posit32es1_bits(sw::universal::tanh(posit32es1_from_bits(a))); }
TVM_DLL uint32_t Posit32es1Cos(uint32_t a) { return posit32es1_bits(sw::universal::cos(posit32es1_from_bits(a))); }
TVM_DLL uint32_t Posit32es1Sin(uint32_t a) { return posit32es1_bits(sw::universal::sin(posit32es1_from_bits(a))); }
TVM_DLL uint32_t Posit32es1Tan(uint32_t a) { return posit32es1_bits(sw::universal::tan(posit32es1_from_bits(a))); }
TVM_DLL uint32_t Posit32es1Erf(uint32_t a) { return posit32es1_bits(sw::universal::erf(posit32es1_from_bits(a))); }
TVM_DLL uint32_t Posit32es1Softmax(uint32_t a) {
  auto p = posit32es1_from_bits(a);
  auto exp_p = sw::universal::exp(p);
  return posit32es1_bits(exp_p / exp_p);
}

// ----- posit8 es1: uint8_t API -----
TVM_DLL uint8_t FloatToPosit8es1(float in) {
  sw::universal::posit<8, 1> p(in);
  return posit8e1_bits(p);
}
TVM_DLL float Posit8es1ToFloat(uint8_t in) { return static_cast<float>(posit8e1_from_bits(in)); }
TVM_DLL uint8_t Posit8es1Add(uint8_t a, uint8_t b) { return posit8e1_bits(posit8e1_from_bits(a) + posit8e1_from_bits(b)); }
TVM_DLL uint8_t Posit8es1Sub(uint8_t a, uint8_t b) { return posit8e1_bits(posit8e1_from_bits(a) - posit8e1_from_bits(b)); }
TVM_DLL uint8_t Posit8es1Mul(uint8_t a, uint8_t b) { return posit8e1_bits(posit8e1_from_bits(a) * posit8e1_from_bits(b)); }
TVM_DLL uint8_t Posit8es1Div(uint8_t a, uint8_t b) { return posit8e1_bits(posit8e1_from_bits(a) / posit8e1_from_bits(b)); }

TVM_DLL uint8_t Posit8es1Max(uint8_t a, uint8_t b) {
  auto pa = posit8e1_from_bits(a), pb = posit8e1_from_bits(b);
  return posit8e1_bits(sw::universal::max(pa, pb));
}
TVM_DLL uint8_t Posit8es1Min(uint8_t a, uint8_t b) {
  auto pa = posit8e1_from_bits(a), pb = posit8e1_from_bits(b);
  return posit8e1_bits(sw::universal::min(pa, pb));
}
TVM_DLL uint8_t Posit8es1Sqrt(uint8_t a) { return posit8e1_bits(sw::universal::sqrt(posit8e1_from_bits(a))); }
TVM_DLL uint8_t Posit8es1Exp(uint8_t a) { return posit8e1_bits(sw::universal::exp(posit8e1_from_bits(a))); }
TVM_DLL uint8_t Posit8es1Log(uint8_t a) { return posit8e1_bits(sw::universal::log(posit8e1_from_bits(a))); }
TVM_DLL uint8_t Posit8es1Pow(uint8_t a, uint8_t b) {
  return posit8e1_bits(sw::universal::pow(posit8e1_from_bits(a), posit8e1_from_bits(b)));
}
TVM_DLL uint8_t Posit8es1Sigmoid(uint8_t a) {
  auto p = posit8e1_from_bits(a);
  return posit8e1_bits(sw::universal::posit<8, 1>(1.0) / (sw::universal::posit<8, 1>(1.0) + sw::universal::exp(-p)));
}
TVM_DLL uint8_t Posit8es1Tanh(uint8_t a) { return posit8e1_bits(sw::universal::tanh(posit8e1_from_bits(a))); }
TVM_DLL uint8_t Posit8es1Cos(uint8_t a) { return posit8e1_bits(sw::universal::cos(posit8e1_from_bits(a))); }
TVM_DLL uint8_t Posit8es1Sin(uint8_t a) { return posit8e1_bits(sw::universal::sin(posit8e1_from_bits(a))); }
TVM_DLL uint8_t Posit8es1Tan(uint8_t a) { return posit8e1_bits(sw::universal::tan(posit8e1_from_bits(a))); }
TVM_DLL uint8_t Posit8es1Erf(uint8_t a) { return posit8e1_bits(sw::universal::erf(posit8e1_from_bits(a))); }
TVM_DLL uint8_t Posit8es1Softmax(uint8_t a) {
  auto p = posit8e1_from_bits(a);
  auto exp_p = sw::universal::exp(p);
  return posit8e1_bits(exp_p / exp_p);
}

TVM_DLL void Posit16es1QuireMatmul(
    uint16_t* A, int64_t M, int64_t K,
    uint16_t* B, int64_t K2, int64_t N,
    uint16_t* C) {
  // Verify dimensions
  if (K != K2) return;
  
  // Tile sizes optimized for L1/L2 cache
  // Posit16 = 2 bytes, quire~18 bytes overhead per accumulator
  // L1 ~32KB, L2 ~256KB
  constexpr int64_t TILE_M = 32;  // Rows of A/C per tile
  constexpr int64_t TILE_N = 64;  // Cols of B/C per tile  
  constexpr int64_t TILE_K = 128; // Reduction dimension tile
  
  // Initialize output to zero
  std::memset(C, 0, M * N * sizeof(uint16_t));
  
  // Allocate quire accumulators for a tile of C
  // Using thread-local storage for parallel execution
  #pragma omp parallel
  {
    // Each thread has its own tile of quire accumulators
    std::vector<sw::universal::quire<16, 1, 30>> q_tile(TILE_M * TILE_N);
    
    // Tiled matrix multiplication with i-k-j loop order
    #pragma omp for collapse(2) schedule(dynamic)
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
      for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
        int64_t i_end = std::min(i0 + TILE_M, M);
        int64_t j_end = std::min(j0 + TILE_N, N);
        int64_t tile_m = i_end - i0;
        int64_t tile_n = j_end - j0;
        
        // Initialize quire accumulators for this C tile
        for (int64_t ti = 0; ti < tile_m; ++ti) {
          for (int64_t tj = 0; tj < tile_n; ++tj) {
            q_tile[ti * TILE_N + tj].clear();
          }
        }
        
        // Process K dimension in tiles
        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
          int64_t k_end = std::min(k0 + TILE_K, K);
          
          // Inner computation: accumulate A_tile @ B_tile into quires
          // Loop order: i-k-j for better cache locality
          for (int64_t i = i0; i < i_end; ++i) {
            int64_t ti = i - i0;
            
            // Process k dimension - A[i,k] is accessed sequentially
            for (int64_t k = k0; k < k_end; ++k) {
              auto pa = posit16es1_from_bits(A[i * K + k]);
              
              // Process j dimension - this inner loop benefits from k being fixed
              for (int64_t j = j0; j < j_end; ++j) {
                int64_t tj = j - j0;
                auto pb = posit16es1_from_bits(B[k * N + j]);
                auto product = pa * pb;
                q_tile[ti * TILE_N + tj] += product.to_value();
              }
            }
          }
        }
        
        // Finalize: convert quires to posits and store
        for (int64_t i = i0; i < i_end; ++i) {
          int64_t ti = i - i0;
          for (int64_t j = j0; j < j_end; ++j) {
            int64_t tj = j - j0;
            sw::universal::posit<16, 1> result;
            sw::universal::convert(q_tile[ti * TILE_N + tj].to_value(), result);
            C[i * N + j] = posit16es1_bits(result);
          }
        }
      }
    }
  }
}

// // ============================================================================
// // Posit16 es2 TRUE Quire Matrix Multiplication
// // ============================================================================
// // C[i,j] = sum(A[i,k] * B[k,j] for k in 0..K) with only ONE rounding per output element.
// TVM_DLL void Posit16es2QuireMatmul(
//     uint16_t* A, int64_t M, int64_t K,
//     uint16_t* B, int64_t K2, int64_t N,
//     uint16_t* C) {
//   // Verify dimensions
//   if (K != K2) return;
  
//   constexpr int64_t TILE_M = 32;
//   constexpr int64_t TILE_N = 64;
//   constexpr int64_t TILE_K = 128;
  
//   std::memset(C, 0, M * N * sizeof(uint16_t));
  
//   #pragma omp parallel
//   {
//     std::vector<sw::universal::quire<16, 2, 30>> q_tile(TILE_M * TILE_N);
    
//     #pragma omp for collapse(2) schedule(dynamic)
//     for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
//       for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
//         int64_t i_end = std::min(i0 + TILE_M, M);
//         int64_t j_end = std::min(j0 + TILE_N, N);
//         int64_t tile_m = i_end - i0;
//         int64_t tile_n = j_end - j0;
        
//         for (int64_t ti = 0; ti < tile_m; ++ti) {
//           for (int64_t tj = 0; tj < tile_n; ++tj) {
//             q_tile[ti * TILE_N + tj].clear();
//           }
//         }
        
//         for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
//           int64_t k_end = std::min(k0 + TILE_K, K);
          
//           for (int64_t i = i0; i < i_end; ++i) {
//             int64_t ti = i - i0;
            
//             for (int64_t k = k0; k < k_end; ++k) {
//               auto pa = posit16_from_bits(A[i * K + k]);
              
//               for (int64_t j = j0; j < j_end; ++j) {
//                 int64_t tj = j - j0;
//                 auto pb = posit16_from_bits(B[k * N + j]);
//                 auto product = pa * pb;
//                 q_tile[ti * TILE_N + tj] += product.to_value();
//               }
//             }
//           }
//         }
        
//         for (int64_t i = i0; i < i_end; ++i) {
//           int64_t ti = i - i0;
//           for (int64_t j = j0; j < j_end; ++j) {
//             int64_t tj = j - j0;
//             sw::universal::posit<16, 2> result;
//             sw::universal::convert(q_tile[ti * TILE_N + tj].to_value(), result);
//             C[i * N + j] = posit16_bits(result);
//           }
//         }
//       }
//     }
//   }
// }

// // ============================================================================
// // Posit32 es1 TRUE Quire Matrix Multiplication
// // ============================================================================
// TVM_DLL void Posit32es1QuireMatmul(
//     uint32_t* A, int64_t M, int64_t K,
//     uint32_t* B, int64_t K2, int64_t N,
//     uint32_t* C) {
//   if (K != K2) return;
  
//   constexpr int64_t TILE_M = 16;
//   constexpr int64_t TILE_N = 32;
//   constexpr int64_t TILE_K = 64;
  
//   std::memset(C, 0, M * N * sizeof(uint32_t));
  
//   #pragma omp parallel
//   {
//     std::vector<sw::universal::quire<32, 1, 30>> q_tile(TILE_M * TILE_N);
    
//     #pragma omp for collapse(2) schedule(dynamic)
//     for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
//       for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
//         int64_t i_end = std::min(i0 + TILE_M, M);
//         int64_t j_end = std::min(j0 + TILE_N, N);
//         int64_t tile_m = i_end - i0;
//         int64_t tile_n = j_end - j0;
        
//         for (int64_t ti = 0; ti < tile_m; ++ti) {
//           for (int64_t tj = 0; tj < tile_n; ++tj) {
//             q_tile[ti * TILE_N + tj].clear();
//           }
//         }
        
//         for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
//           int64_t k_end = std::min(k0 + TILE_K, K);
          
//           for (int64_t i = i0; i < i_end; ++i) {
//             int64_t ti = i - i0;
            
//             for (int64_t k = k0; k < k_end; ++k) {
//               auto pa = posit32es1_from_bits(A[i * K + k]);
              
//               for (int64_t j = j0; j < j_end; ++j) {
//                 int64_t tj = j - j0;
//                 auto pb = posit32es1_from_bits(B[k * N + j]);
//                 auto product = pa * pb;
//                 q_tile[ti * TILE_N + tj] += product.to_value();
//               }
//             }
//           }
//         }
        
//         for (int64_t i = i0; i < i_end; ++i) {
//           int64_t ti = i - i0;
//           for (int64_t j = j0; j < j_end; ++j) {
//             int64_t tj = j - j0;
//             sw::universal::posit<32, 1> result;
//             sw::universal::convert(q_tile[ti * TILE_N + tj].to_value(), result);
//             C[i * N + j] = posit32es1_bits(result);
//           }
//         }
//       }
//     }
//   }
// }

// // ============================================================================
// // Posit32 es2 TRUE Quire Matrix Multiplication
// // ============================================================================
// TVM_DLL void Posit32es2QuireMatmul(
//     uint32_t* A, int64_t M, int64_t K,
//     uint32_t* B, int64_t K2, int64_t N,
//     uint32_t* C) {
//   if (K != K2) return;
  
//   constexpr int64_t TILE_M = 16;
//   constexpr int64_t TILE_N = 32;
//   constexpr int64_t TILE_K = 64;
  
//   std::memset(C, 0, M * N * sizeof(uint32_t));
  
//   #pragma omp parallel
//   {
//     std::vector<sw::universal::quire<32, 2, 30>> q_tile(TILE_M * TILE_N);
    
//     #pragma omp for collapse(2) schedule(dynamic)
//     for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
//       for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
//         int64_t i_end = std::min(i0 + TILE_M, M);
//         int64_t j_end = std::min(j0 + TILE_N, N);
//         int64_t tile_m = i_end - i0;
//         int64_t tile_n = j_end - j0;
        
//         for (int64_t ti = 0; ti < tile_m; ++ti) {
//           for (int64_t tj = 0; tj < tile_n; ++tj) {
//             q_tile[ti * TILE_N + tj].clear();
//           }
//         }
        
//         for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
//           int64_t k_end = std::min(k0 + TILE_K, K);
          
//           for (int64_t i = i0; i < i_end; ++i) {
//             int64_t ti = i - i0;
            
//             for (int64_t k = k0; k < k_end; ++k) {
//               auto pa = posit32_from_bits(A[i * K + k]);
              
//               for (int64_t j = j0; j < j_end; ++j) {
//                 int64_t tj = j - j0;
//                 auto pb = posit32_from_bits(B[k * N + j]);
//                 auto product = pa * pb;
//                 q_tile[ti * TILE_N + tj] += product.to_value();
//               }
//             }
//           }
//         }
        
//         for (int64_t i = i0; i < i_end; ++i) {
//           int64_t ti = i - i0;
//           for (int64_t j = j0; j < j_end; ++j) {
//             int64_t tj = j - j0;
//             sw::universal::posit<32, 2> result;
//             sw::universal::convert(q_tile[ti * TILE_N + tj].to_value(), result);
//             C[i * N + j] = posit32_bits(result);
//           }
//         }
//       }
//     }
//   }
// }

// // ============================================================================
// // Posit8 es1 TRUE Quire Matrix Multiplication
// // ============================================================================
// TVM_DLL void Posit8es1QuireMatmul(
//     uint8_t* A, int64_t M, int64_t K,
//     uint8_t* B, int64_t K2, int64_t N,
//     uint8_t* C) {
//   if (K != K2) return;
  
//   constexpr int64_t TILE_M = 64;
//   constexpr int64_t TILE_N = 128;
//   constexpr int64_t TILE_K = 256;
  
//   std::memset(C, 0, M * N * sizeof(uint8_t));
  
//   #pragma omp parallel
//   {
//     std::vector<sw::universal::quire<8, 1, 10>> q_tile(TILE_M * TILE_N);
    
//     #pragma omp for collapse(2) schedule(dynamic)
//     for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
//       for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
//         int64_t i_end = std::min(i0 + TILE_M, M);
//         int64_t j_end = std::min(j0 + TILE_N, N);
//         int64_t tile_m = i_end - i0;
//         int64_t tile_n = j_end - j0;
        
//         for (int64_t ti = 0; ti < tile_m; ++ti) {
//           for (int64_t tj = 0; tj < tile_n; ++tj) {
//             q_tile[ti * TILE_N + tj].clear();
//           }
//         }
        
//         for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
//           int64_t k_end = std::min(k0 + TILE_K, K);
          
//           for (int64_t i = i0; i < i_end; ++i) {
//             int64_t ti = i - i0;
            
//             for (int64_t k = k0; k < k_end; ++k) {
//               auto pa = posit8e1_from_bits(A[i * K + k]);
              
//               for (int64_t j = j0; j < j_end; ++j) {
//                 int64_t tj = j - j0;
//                 auto pb = posit8e1_from_bits(B[k * N + j]);
//                 auto product = pa * pb;
//                 q_tile[ti * TILE_N + tj] += product.to_value();
//               }
//             }
//           }
//         }
        
//         for (int64_t i = i0; i < i_end; ++i) {
//           int64_t ti = i - i0;
//           for (int64_t j = j0; j < j_end; ++j) {
//             int64_t tj = j - j0;
//             sw::universal::posit<8, 1> result;
//             sw::universal::convert(q_tile[ti * TILE_N + tj].to_value(), result);
//             C[i * N + j] = posit8e1_bits(result);
//           }
//         }
//       }
//     }
//   }
// }

// // ============================================================================
// // Posit8 es2 TRUE Quire Matrix Multiplication
// // ============================================================================
// TVM_DLL void Posit8es2QuireMatmul(
//     uint8_t* A, int64_t M, int64_t K,
//     uint8_t* B, int64_t K2, int64_t N,
//     uint8_t* C) {
//   if (K != K2) return;
  
//   constexpr int64_t TILE_M = 64;
//   constexpr int64_t TILE_N = 128;
//   constexpr int64_t TILE_K = 256;
  
//   std::memset(C, 0, M * N * sizeof(uint8_t));
  
//   #pragma omp parallel
//   {
//     std::vector<sw::universal::quire<8, 2, 10>> q_tile(TILE_M * TILE_N);
    
//     #pragma omp for collapse(2) schedule(dynamic)
//     for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
//       for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
//         int64_t i_end = std::min(i0 + TILE_M, M);
//         int64_t j_end = std::min(j0 + TILE_N, N);
//         int64_t tile_m = i_end - i0;
//         int64_t tile_n = j_end - j0;
        
//         for (int64_t ti = 0; ti < tile_m; ++ti) {
//           for (int64_t tj = 0; tj < tile_n; ++tj) {
//             q_tile[ti * TILE_N + tj].clear();
//           }
//         }
        
//         for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
//           int64_t k_end = std::min(k0 + TILE_K, K);
          
//           for (int64_t i = i0; i < i_end; ++i) {
//             int64_t ti = i - i0;
            
//             for (int64_t k = k0; k < k_end; ++k) {
//               auto pa = posit8_from_bits(A[i * K + k]);
              
//               for (int64_t j = j0; j < j_end; ++j) {
//                 int64_t tj = j - j0;
//                 auto pb = posit8_from_bits(B[k * N + j]);
//                 auto product = pa * pb;
//                 q_tile[ti * TILE_N + tj] += product.to_value();
//               }
//             }
//           }
//         }
        
//         for (int64_t i = i0; i < i_end; ++i) {
//           int64_t ti = i - i0;
//           for (int64_t j = j0; j < j_end; ++j) {
//             int64_t tj = j - j0;
//             sw::universal::posit<8, 2> result;
//             sw::universal::convert(q_tile[ti * TILE_N + tj].to_value(), result);
//             C[i * N + j] = posit8_bits(result);
//           }
//         }
//       }
//     }
//   }
// }

/**
 * Posit16 es1 QuireMatmul - Row-wise Version (Alternative name for explicit row computation)
 * 
 * @param A_ptr   Pointer to matrix A (M x K), row-major
 * @param row     Row index to compute (0 <= row < M)
 * @param K       Shared dimension (columns of A, rows of B)
 * @param B_ptr   Pointer to matrix B (K x N), row-major
 * @param N       Number of columns in B and C
 * @param C_ptr   Pointer to output matrix C (M x N), row-major
 */
TVM_DLL void Posit16es1QuireMatmulRow(
    uint16_t* A_ptr, int64_t row, int64_t K,
    uint16_t* B_ptr, int64_t N,
    uint16_t* C_ptr) {
  
  // For each output element C[row, j], compute dot product using quire
  // #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int64_t j = 0; j < N; ++j) {
    sw::universal::quire<16, 1, 30> q;
    q.clear();
    
    // Accumulate A[row, k] * B[k, j] in quire (NO intermediate rounding)
    for (int64_t k = 0; k < K; ++k) {
      auto pa = posit16es1_from_bits(A_ptr[row * K + k]);
      auto pb = posit16es1_from_bits(B_ptr[k * N + j]);
      q += (pa * pb).to_value();
    }
    
    // Convert quire to posit (SINGLE rounding per output element)
    sw::universal::posit<16, 1> result;
    sw::universal::convert(q.to_value(), result);
    C_ptr[row * N + j] = posit16es1_bits(result);
  }
}

// ============================================================================
// Posit16 es1 Quire Matrix Multiplication - Single Element
// ============================================================================
// Compute single output element C[offset] = A[0:K] dot B[col, 0:K] with quire accumulation
// This is designed for TIR parallel loops where each thread computes one output element
TVM_DLL void Posit16es1QuireMatmulElem(
    uint16_t* A,       // Input vector A (size K)
    int64_t K,         // Length of vectors
    uint16_t* B,       // Input matrix B (column-major access)
    int64_t col,       // Column index in B to compute dot product with
    int64_t N,         // Number of columns in B (stride)
    uint16_t* C,       // Output array
    int64_t offset     // Offset in C array to write result
) {
  // Use quire for exact accumulation
  sw::universal::quire<16, 1> q;
  q.clear();
  
  // Accumulate A[k] * B[k, col] in quire (NO intermediate rounding)
  for (int64_t k = 0; k < K; ++k) {
    auto pa = posit16es1_from_bits(A[k]);
    auto pb = posit16es1_from_bits(B[k * N + col]);
    q += (pa * pb).to_value();
  }
  
  // Convert quire to posit (SINGLE rounding)
  sw::universal::posit<16, 1> result;
  sw::universal::convert(q.to_value(), result);
  C[offset] = posit16es1_bits(result);
}

// Version with explicit A and B offsets for multidimensional arrays
TVM_DLL void Posit16es1QuireMatmulElemWithOffset(
    uint16_t* A,       // Input array A base pointer
    int64_t a_offset,  // Offset into A array (elements, not bytes)
    int64_t K,         // Length of vectors
    uint16_t* B,       // Input matrix B base pointer
    int64_t b_offset,  // Offset into B array (elements, not bytes)
    int64_t col,       // Column index in B to compute dot product with
    int64_t N,         // Number of columns in B (stride)
    uint16_t* C,       // Output array
    int64_t offset     // Offset in C array to write result
) {
  // Use quire for exact accumulation
  sw::universal::quire<16, 1> q;
  q.clear();
  
  // Accumulate A[a_offset + k] * B[b_offset + k*N + col] in quire
  for (int64_t k = 0; k < K; ++k) {
    auto pa = posit16es1_from_bits(A[a_offset + k]);
    auto pb = posit16es1_from_bits(B[b_offset + k * N + col]);
    q += (pa * pb).to_value();
  }
  
  // Convert quire to posit (SINGLE rounding)
  sw::universal::posit<16, 1> result;
  sw::universal::convert(q.to_value(), result);
  C[offset] = posit16es1_bits(result);
}

}
