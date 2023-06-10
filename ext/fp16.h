#pragma once
#include "rt.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: mf8_t  https://en.wikipedia.org/wiki/Minifloat

// AVX512 support both fp16 and bf16 but only on three (server grade) processors so far
// see:   https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
//        search for bf16 and fp16 with [x] AVX512 and [x] AVX
//        CPUID Flags: AVX512_BF16 + AVX512_FP16 + AVX512VL
//        however it is easy to emulate with absence of bf16/fp16 AVX512 extension
// https://en.wikichip.org/wiki/x86/avx512_bf16

// https://en.wikipedia.org/wiki/Half-precision_floating-point_format

typedef begin_packed struct _fp16_u_ {
    uint16_t bytes;
} end_packed _fp16_t_;

typedef begin_packed struct _bf16_u_ { // "brain fart" "brain float 16 bit"
    uint16_t bytes;
} end_packed _bf16_t_;

#undef fp16_t
#define fp16_t _fp16_t_ // using define allows to write #ifdef fp16_t
#undef bf16_t
#define bf16_t _bf16_t_

static_assert(sizeof(fp16_t) == 2, "fp16_t size must be 2");
static_assert(sizeof(bf16_t) == 2, "fp16_t size must be 2");


#define fp16x(hex)       ((fp16_t){ .bytes = hex })

#define FP16_NAN         fp16x(0x7FFF) // not a number one of many possible values
#define FP16_PINF        fp16x(0x7C00) // positive infinity
#define FP16_NINF        fp16x(0xFC00) // negative infinity
#define F16_DECIMAL_DIG  5             // # of decimal digits of rounding precision
#define F16_DIG          4             // # of decimal digits of precision
#define F16_EPSILON      fp16x(0x1400) // 9.7656250E-04 smallest such that 1.0 + F16_EPSILON != 1.0
#define F16_HAS_SUBNORM  1             // type does support subnormal numbers
#define F16_GUARD        0
#define F16_MANT_DIG     10            // # of bits in mantissa
#define F16_MAX          fp16x(0x7BFF) // 6.5504000E+04
#define F16_MAX_10_EXP   8             // max decimal exponent
#define F16_MAX_EXP      15            // max binary exponent
#define F16_MIN          fp16x(0x0400) // 6.1035156E-05 min normalized positive value
#define F16_MIN_10_EXP   (-8)          // min decimal exponent
#define F16_MIN_EXP      (-14)         // min binary exponent
#define F16_NORMALIZE    0
#define F16_RADIX        2             // exponent radix
#define F16_TRUE_MIN     fp16x(0x0001) // 2.9802322E-08 subnormal pow(2,-24) ~5.96E-8

inline bool fp16_isnan(fp16_t v) { return ((v.bytes >> 10) & 0x1F) == 0x1F && (v.bytes & 0x3FF) != 0; }

inline bool fp16_isfinite(fp16_t v) { return ((v.bytes >> 10) & 0x1F) != 0x1F; }

// Terminology:
// "normal" actually "normalized" as in "shifted left till highest bit of
// mantissa is not zero and than hide that bit and shift is expressed as
// exponent".
// "denormal" "subnormal" actually "not normalized" as in "nothing
// is hidden in mantissa, it is not shifted and exponent is equal to zero.
//
// This allows floating number to have "subnormal" precision close to zero.
// For float32 is actually extends close to zero density by 8 bit of exponent
// and makes minimum strictly positive (subnormal) value pow(2, -31) for fp32_t
// and pow(2, -24) fp16_t.
//
// The magic is based on exponent == 0 is actually pow(2, -exponent_range/2)
// which allows for fp16_t representation of very small density of numbers
// in the 0 to pow(2, -13) interval. The density is pow(2, -24) for 16 bit floats.
// See:
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format

inline fp16_t fp32to16(fp32_t f32) {
	// Float structure:
	// 1-bit sign
	// 8-bit exponent
	// 23-bit mantissa
	// s eeeeeeee mmmmmmmmmmmmmmmmmmmmmmm
	// Different float categories:
	// 0 0000 0000 00000000000000000000000  +zero
	// 1 0000 0000 00000000000000000000000  -zero
	// s 0000 0000 mmmmmmmmmmmmmmmmmmmmmmm  A subnormal number, mmmmm != 0,
    //                                      interpreted as (-1)^s * 2^-126 * 0.mmmmm
	// s eeee eeee xxxxxxxxxxxxxxxxxxxxxxx  A normal number, eeeee != 0,
    //                                      interpreted as (-1)^s * 2^(e-127) * 1.mmmmm
	// 0 1111 1111 00000000000000000000000  +inf
	// 1 1111 1111 00000000000000000000000  -inf
	// y 1111 1111 1xxxxxxxxxxxxxxxxxxxxxx  Quiet NaN, y and xxxxx are arbitrary
    //                                      (custom) payload for the NaN.
	// y 1111 1111 0xxxxxxxxxxxxxxxxxxxxxx  Signalling NaN, y and xxxxx != 0 is
    //                                      arbitrary payload for the NaN.

	// When writing our custom low-precision minifloat, make sure that values
    // in each of the above categories stays in
	// the same category, if possible:
	// subnormals:
    //      conversion gracefully loses precision of mantissa.
	// normals:
    //      conversion with exponent out of range can cause overflow or underflow.
	//      if the exponent is too large, result is +inf/-inf.
	//      If the exponent is too small, the result will be zero.
	// +/- inf, +/- zero: conversion does not affect the value.
	// NaN and signaling NaN: conversion preseres signalling.
	uint32_t v = *(uint32_t*)&f32;
	uint32_t biased_exponent = (v & 0x7F800000) >> 23;
	uint32_t mantissa = v & 0x7FFFFF;
	uint32_t sign = (v & 0x80000000) >> 16;
    uint32_t result;
	const uint32_t maxbiased_exponent = (1 << 5) - 1;
	int true_exponent = biased_exponent - 127;
	int new_biased_exponent;
	// Compute the new biased exponent value to send.
	if (biased_exponent != 0xFF && biased_exponent != 0) {
		new_biased_exponent = true_exponent + 15;
		if (new_biased_exponent >= (int)maxbiased_exponent) {
			new_biased_exponent = maxbiased_exponent;
			mantissa = 0; // inf, not NaN
		} else if (new_biased_exponent < -10) {
            // too small for subnormal
            new_biased_exponent = 0;
            mantissa = 0;
        } else if (new_biased_exponent < 0) {
            assert(0 <= 1 - new_biased_exponent);
            mantissa |= 0x800000;  // add hidden bit
            mantissa >>= (1 - new_biased_exponent);
            new_biased_exponent = 0;
        } else {
        }
        mantissa = (mantissa >> (23 - 10)) & 0x3FF;
        result = sign | (new_biased_exponent << 10) | mantissa;
	} else {
        // either all zeroes (+/-zero or denormal) or all ones (nan or inf).
		new_biased_exponent = biased_exponent & 0x1F;
	    uint32_t new_mantissa = (mantissa >> (23 - 10)) & 0x3FF;
    	// If the float was a signaling NaN, make sure it stays
        // a signaling NaN:
	    if (biased_exponent == 0x1F && mantissa != 0 && new_mantissa == 0) {
	    	new_mantissa = 1; // denotes a NaN but not a signaling NaN
        }
        result = sign | (new_biased_exponent << 10) | new_mantissa;
    }
    assert(0 <= result && result <= UINT16_MAX);
    fp16_t r = { .bytes = (uint16_t)result };
	return r;
}

inline fp32_t fp16to32(fp16_t fp16) {
	uint32_t sign = (fp16.bytes & 0x8000) << 16;
	uint32_t exponent = (fp16.bytes & 0x7C00) >> 10;
	uint32_t mantissa = fp16.bytes & 0x03FF;
    uint32_t result;
    // Shift back the decoded mantissa to proper position.
    if (exponent == 0 && mantissa == 0) {  // +/- zero
        result = sign << 31;
    } else if (exponent == 0x1f && mantissa == 0) {  // +/- infinity
        result = sign | (0xFF << 23);
    } else if (exponent == 0x1f && mantissa != 0) {  // NaN
        // preserve mantisa for signaling NaN (highest bit of mantissa)
        result = sign | (0xFF << 23) | (mantissa << 13);
	} else if (exponent != 0) { // normal exponent is not zero
        exponent = exponent + (127 - 15);
        mantissa = mantissa << 13;
        result = sign | (exponent << 23) | mantissa;
    } else if (mantissa != 0) { // not zero and not normalized
        assert(exponent == 0);
        exponent = 127 - 15;
        while ((mantissa & 0x00000400) == 0) {  // left shift mantissa
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa <<= 13;
        exponent++;
        mantissa &= 0x7FFFFF; // remove hidden bit
        result = sign | (exponent << 23) | mantissa;
//      println("normalized result: 0x%08X", result);
    } else {
        assert(mantissa == 0 && exponent == 0, "+/- zero");
        result = sign;
    }
	return *(fp32_t*)&result;
}

inline fp16_t fp16_add(fp16_t x, fp16_t y) {
    return fp32to16(fp16to32(x) + fp16to32(y));
}

inline fp16_t fp16_sub(fp16_t x, fp16_t y) {
    return fp32to16(fp16to32(x) - fp16to32(y));
}
inline fp16_t fp16_mul(fp16_t x, fp16_t y) {
    return fp32to16(fp16to32(x) * fp16to32(y));
}
inline fp16_t fp16_div(fp16_t x, fp16_t y) {
    return fp32to16(fp16to32(x) / fp16to32(y));
}

inline int fp16_compare(fp16_t x, fp16_t y) {
    fp16_t diff = fp16_sub(x, y);
    return (diff.bytes & 0x8000) ? -1 : (diff.bytes == 0) ? 0 : +1;
}

inline bool fp16_equ(fp16_t x, fp16_t y) { return fp16_compare(x, y) == 0; }
inline bool fp16_leq(fp16_t x, fp16_t y) { return fp16_compare(x, y) <= 0; }
inline bool fp16_les(fp16_t x, fp16_t y) { return fp16_compare(x, y) <  0; }
inline bool fp16_gtr(fp16_t x, fp16_t y) { return fp16_compare(x, y) >  0; }
inline bool fp16_gte(fp16_t x, fp16_t y) { return fp16_compare(x, y) >= 0; }
inline bool fp16_neq(fp16_t x, fp16_t y) { return fp16_compare(x, y) != 0; }

inline fp32_t bf16to32(const bf16_t bf16) {
    const uint32_t uint32 = ((uint32_t)(bf16.bytes)) << 16;
    return *(fp32_t*)&uint32;
}

inline bf16_t bf32to16(const fp32_t fp32) {
    const uint16_t uint16 = (uint16_t)((*(uint32_t*)&fp32) >> 16);
    return *(bf16_t*)&uint16;
}

inline bf16_t bf16_add(bf16_t x, bf16_t y) {
    return bf32to16(bf16to32(x) + bf16to32(y));
}

inline bf16_t bf16_sub(bf16_t x, bf16_t y) {
    return bf32to16(bf16to32(x) - bf16to32(y));
}
inline bf16_t bf16_mul(bf16_t x, bf16_t y) {
    return bf32to16(bf16to32(x) * bf16to32(y));
}
inline bf16_t bf16_div(bf16_t x, bf16_t y) {
    return bf32to16(bf16to32(x) / bf16to32(y));
}

inline int bf16_compare(bf16_t x, bf16_t y) {
    bf16_t diff = bf16_sub(x, y);
    return (diff.bytes & 0x8000) ? -1 : (diff.bytes == 0) ? 0 : +1;
}

inline bool bf16_equ(bf16_t x, bf16_t y) { return bf16_compare(x, y) == 0; }
inline bool bf16_leq(bf16_t x, bf16_t y) { return bf16_compare(x, y) <= 0; }
inline bool bf16_les(bf16_t x, bf16_t y) { return bf16_compare(x, y) <  0; }
inline bool bf16_gtr(bf16_t x, bf16_t y) { return bf16_compare(x, y) >  0; }
inline bool bf16_gte(bf16_t x, bf16_t y) { return bf16_compare(x, y) >= 0; }
inline bool bf16_neq(bf16_t x, bf16_t y) { return bf16_compare(x, y) != 0; }

#ifdef RT_IMPLEMENTATION

#ifdef FP16_TESTS

void fp16_test() {
#if 0
    #define dump_f16(label, v) println("%-35s 0x%04X %.7E", label, v.bytes, fp16to32(v))
    #define dump_f32(label, v) println("%-35s 0x%08X %.7E", label, *(uint32_t*)&v, v)
#else
    #define dump_f16(label, v) (void)(v); // unused
    #define dump_f32(label, v) (void)(v); // unused
#endif
    fp16_t one = fp32to16(1.0f);
    assert(fp16_equ(one, fp16x(0x3C00)));
    fp16_t two = fp32to16(2.0);
    assert(fp16_equ(two, fp16_add(one, one)));
    fp16_t nan   = FP16_NAN;
    fp16_t inf_p = FP16_PINF;
    fp16_t inf_n = FP16_NINF;
    assert(fp16_isnan(nan));
    assert(!fp16_isfinite(inf_p));
    assert(!fp16_isfinite(inf_n));
    assert( fp16_isfinite(F16_MAX));
    assert( fp16_isfinite(F16_MIN));
    assert( fp16_isfinite(F16_EPSILON));
    assert( fp16_isfinite(F16_TRUE_MIN));
    {
        fp16_t small = F16_MIN;
        fp16_t a;
        for (;;) {
            a = fp16_add(small, F16_MAX);
            if (!fp16_isfinite(a) || !fp16_equ(a, F16_MAX)) { break; }
            small = fp16_add(small, small);
        }
        assert(!fp16_isfinite(a), "should not be able to add anything to MAX");
        assert( fp16_equ(small, fp32to16(32)), "density between: 32768 65519 is 32");
        dump_f16("a    :", a);
        dump_f16("small:", small);
    }
    {   // verify epsilon
        fp16_t f16_epsilon = one;
        fp16_t last = one;
        while (fp16_neq(one, fp16_add(one, f16_epsilon))) {
            last = f16_epsilon;
            f16_epsilon = fp16_div(f16_epsilon, fp32to16(2.0));
        }
        f16_epsilon = last;
        assert(f16_epsilon.bytes == F16_EPSILON.bytes);
    }
    {
        float f32e_add_1 = 1.0f + fp16to32(F16_EPSILON);
        assert(f32e_add_1 != 1.0f);
        fp16_t f16e_add_1 = fp32to16(f32e_add_1);
        assert(f16e_add_1.bytes != one.bytes);
    }
    fp16_t one_plus_epsilon = fp16_add(one, F16_EPSILON);
    assert(fp16_neq(one_plus_epsilon, one));
    fp16_t one_plus_min = fp16_add(one, F16_MIN);
    assert(fp16_equ(one_plus_min, one));
    fp16_t smallest_positive_subnormal_number   = fp16x(0x0001); // fp32to16(0.000000059604645f);
    fp16_t largest_subnormal_number             = fp16x(0x03ff); // fp32to16(0.000060975552f);
    fp16_t smallest_positive_normal_number      = fp16x(0x0400); // fp32to16(0.00006103515625f);
    fp16_t nearest_value_to_one_third           = fp16x(0x3555); // fp32to16(0.33325195f);
    fp16_t largest_less_than_one                = fp16x(0x3BFF); // fp32to16(0.99951172f);
    fp16_t smallest_number_larger_than_one      = fp16x(0x3c01); // fp32to16(1.00097656f);
    fp16_t largest_normal_number                = fp16x(0x7BFF); // fp32to16(65504.0f);
    fp16_t largest_normal_number_negative       = fp16x(0xFBFF); // fp32to16(-65504.0f);

    const float fp16_minimum_positive_subnormal_value_as_fp32 =
        (float)pow(2,-24);
    assert(fp16to32(fp32to16(fp16_minimum_positive_subnormal_value_as_fp32)) ==
           fp16_minimum_positive_subnormal_value_as_fp32);
    assert(fp16to32(fp32to16(fp16_minimum_positive_subnormal_value_as_fp32)) ==
           fp16_minimum_positive_subnormal_value_as_fp32);
    const float fp16_minimum_positive_normal_value_as_fp32 = (float)pow(2,-14);
    assert(fp16to32(fp32to16(fp16_minimum_positive_normal_value_as_fp32)) ==
           fp16_minimum_positive_normal_value_as_fp32);
    // (2-2^-10) * 2^15 = 65504
    const float f16_maximum_representable_value_as_fp32 =
        (float)((2 - pow(2, -10)) * pow(2, 15));
    assert(fp16to32(fp32to16(f16_maximum_representable_value_as_fp32)) ==
           f16_maximum_representable_value_as_fp32);
    dump_f16("smallest_positive_subnormal_number", smallest_positive_subnormal_number);
    dump_f16("largest_subnormal_number          ", largest_subnormal_number);
    dump_f16("smallest_positive_normal_number   ", smallest_positive_normal_number);
    dump_f16("nearest_value_to_one_third        ", nearest_value_to_one_third);
    dump_f16("largest_less_than_one             ", largest_less_than_one);
    dump_f16("smallest_number_larger_than_one   ", smallest_number_larger_than_one);
    dump_f16("largest_normal_number             ", largest_normal_number);
    dump_f16("largest_normal_number_negative    ", largest_normal_number_negative);
    dump_f16("F16_EPSILON                       ", F16_EPSILON);
    dump_f16("F16_MAX                           ", F16_MAX);
    dump_f16("F16_MIN                           ", F16_MIN);
    dump_f16("F16_TRUE_MIN                      ", F16_TRUE_MIN);
    static uint32_t seed;
    static fp16_t a[10];
    static fp16_t b[countof(a)];
    fp16_t sum = fp32to16(0);
    fp32_t sum32 = 0;
    fp64_t sum64 = 0;
    for (int i = 0; i < countof(a); i++) {
        a[i] = fp32to16((float)i);
        b[i] = fp32to16((float)(countof(a) - 1 - i));
        sum32 += (float)i + (float)i;
        sum64 += i + i;
        println("[%d] a:%.6f b:%.6f", i, fp16to32(a[i]), fp16to32(b[i]));
        sum = fp16_add(fp16_add(a[i], b[i]), sum);
    }
    println("sum fp16:%.6f fp32:%.6f fp64:%.6f", fp16to32(sum), sum32, sum64);

    for (int i = 0; i < countof(a); i++) {
        double ai = 2.0 * ((random32(&seed) / (double)UINT32_MAX) - 1.0);
        a[i] = fp32to16((float)ai);
        double bi = 2.0 * ((random32(&seed) / (double)UINT32_MAX) - 1.0);
        b[i] = fp32to16((float)bi);
        println("%.6f %.6f", fp16to32(a[i]), fp16to32(b[i]));
        sum32 += (float)ai * (float)bi;
        sum64 += ai * bi;
    }
    for (int i = 0; i < countof(a); i++) {
        sum = fp16_add(fp16_mul(a[i], b[i]), sum);
    }
    println("sum fp16:%.6f fp32:%.6f fp64:%.6f", fp16to32(sum), sum32, sum64);
    sum = fp32to16(0);
    // all integers:
    for (int i = 0; i < (2U << 10); i++) {
        fp32_t f32i = (float)(i);
        fp16_t f16i = fp32to16(f32i);
        assert(fp16to32(f16i) == f32i);
    }
    // all exact fractions:
    for (int i = 0; i < (1 << 10); i++) {
        // "The stored exponents 0b00000 and 0b11111 are interpreted specially."
        for (int exp = 1; exp < 0xF; exp++) {
            // deminishing values toward zero both "normalized" and "not normalized"
            fp32_t f32i = (float)(i) / (float)(1 << exp);
            const float minimum_positive_subnormal_value = (float)pow(2,-24);
            assert(f32i == 0 || fabs(f32i) >= minimum_positive_subnormal_value);
            if (exp <= 10) { // normal (aka "normalized")
                const float minimum_positive_normal_value = (float)pow(2,-14);
                assert(f32i == 0 || fabs(f32i) >= minimum_positive_normal_value);
            }
            fp16_t f16i = fp32to16(f32i);
            assert(fp16to32(f16i) == f32i, "fp16: %.7e fp32: %.7e", fp16to32(f16i), f32i);
        }
    }
    for (int i = 0; i < (1 << 10); i++) {
        // "The stored exponents 0b00000 and 0b11111 are interpreted specially."
        for (int exp = 1; exp < 0xF; exp++) {
            // toward maximum
            fp32_t f32i = (float)(i) / (float)(1 << 10) * (1 << exp);
            // (2-2^-10) * 2^15 = 65504
            const float maximum_representable_value = (float)((2 - pow(2, -10)) * pow(2, 15));
            assert(fabs(f32i) <= maximum_representable_value);
            fp16_t f16i = fp32to16(f32i);
            assert(fp16to32(f16i) == f32i);
        }
    }
}

#endif // FP16_TESTS

#endif // RT_IMPLEMENTATION

#ifdef __cplusplus
} // extern "C"
#endif
