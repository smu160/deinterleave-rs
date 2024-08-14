#![forbid(unsafe_code)]
#![feature(portable_simd, avx512_target_feature)]

use bytemuck::cast_slice;
use num_complex::Complex;
use std::simd::{f32x16, f64x8, simd_swizzle, Simd, SimdElement};

macro_rules! impl_separate_re_im {
    ($func_name:ident, $precision:ty, $lanes:literal, $simd_vec:ty) => {
        /// Utility function to separate interleaved format signals (i.e., Vector of Complex Number Structs)
        /// into separate vectors for the corresponding real and imaginary components.
        #[multiversion::multiversion(
                                    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                            "x86_64+avx2+fma", // x86_64-v3
                                            "x86_64+sse4.2", // x86_64-v2
                                            "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                            "x86+avx2+fma",
                                            "x86+sse4.2",
                                            "x86+sse2",
                                        ))]
        #[inline]
        pub fn $func_name(
            signal: &[Complex<$precision>],
        ) -> (Vec<$precision>, Vec<$precision>) {
            let n = signal.len();
            let mut reals = vec![0.0; n];
            let mut imags = vec![0.0; n];

            let complex_t: &[$precision] = cast_slice(signal);
            const CHUNK_SIZE: usize = $lanes * 2;

            for ((chunk, chunk_re), chunk_im) in complex_t
                .chunks_exact(CHUNK_SIZE)
                .zip(reals.chunks_exact_mut($lanes))
                .zip(imags.chunks_exact_mut($lanes))
            {
                let (first_half, second_half) = chunk.split_at($lanes);

                let a = <$simd_vec>::from_slice(&first_half);
                let b = <$simd_vec>::from_slice(&second_half);
                let (re_deinterleaved, im_deinterleaved) = a.deinterleave(b);

                chunk_re.copy_from_slice(&re_deinterleaved.to_array());
                chunk_im.copy_from_slice(&im_deinterleaved.to_array());
            }

            let remainder = complex_t.chunks_exact(CHUNK_SIZE).remainder();
            let reals_rem = reals.chunks_exact_mut($lanes).into_remainder();
            let imags_rem = imags.chunks_exact_mut($lanes).into_remainder();

            remainder
                .chunks_exact(2)
                .zip(reals_rem.iter_mut())
                .zip(imags_rem.iter_mut())
                .for_each(|((c, re), im)| {
                    *re = c[0];
                    *im = c[1];
                });


            (reals, imags)
        }
    };
}

impl_separate_re_im!(simd_deinterleave_32, f32, 16, f32x16);
impl_separate_re_im!(simd_deinterleave_64, f64, 8, f64x8);

#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
#[inline]
pub fn deinterleave_autovec<T: Copy + Default>(input: &[T]) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;

    let out_len = input.len() / 2;
    let mut out_odd = vec![T::default(); out_len];
    let mut out_even = vec![T::default(); out_len];

    input
        .chunks_exact(CHUNK_SIZE * 2)
        .zip(out_odd.chunks_exact_mut(CHUNK_SIZE))
        .zip(out_even.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|((in_chunk, odds), evens)| {
            odds[0] = in_chunk[0];
            evens[0] = in_chunk[1];
            odds[1] = in_chunk[2];
            evens[1] = in_chunk[3];
            odds[2] = in_chunk[4];
            evens[2] = in_chunk[5];
            odds[3] = in_chunk[6];
            evens[3] = in_chunk[7];
        });

    (out_odd, out_even)
}

#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
#[inline]
pub fn deinterleave_simd_swizzle_x86_64_v4<T: Copy + Default + SimdElement>(
    input: &[T],
) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;
    const DOUBLE_CHUNK: usize = CHUNK_SIZE * 2;

    let out_len = input.len() / 2;
    let mut out_odd = vec![T::default(); out_len];
    let mut out_even = vec![T::default(); out_len];

    input
        .chunks_exact(DOUBLE_CHUNK)
        .zip(out_odd.chunks_exact_mut(CHUNK_SIZE))
        .zip(out_even.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|((in_chunk, odds), evens)| {
            let in_simd: Simd<T, DOUBLE_CHUNK> = Simd::from_array(in_chunk.try_into().unwrap());
            let result = simd_swizzle!(in_simd, [0, 2, 4, 6, 1, 3, 5, 7]);
            let result_arr = result.to_array();
            odds.copy_from_slice(&result_arr[..CHUNK_SIZE]);
            evens.copy_from_slice(&result_arr[CHUNK_SIZE..]);
        });

    // Process the remainder, too small for the vectorized loop
    let input_rem = input.chunks_exact(DOUBLE_CHUNK).remainder();
    let odds_rem = out_odd.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    let evens_rem = out_even.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    input_rem
        .chunks_exact(2)
        .zip(odds_rem.iter_mut())
        .zip(evens_rem.iter_mut())
        .for_each(|((inp, odd), even)| {
            *odd = inp[0];
            *even = inp[1];
        });
    (out_odd, out_even)
}

// We don't multiversion for AVX-512 here and keep the chunk size below AVX-512
// because we haven't seen any gains from it in benchmarks.
// This might be due to us running benchmarks on Zen4 which implements AVX-512
// on top of 256-bit wide execution units.
//
// If benchmarks on "real" AVX-512 show improvement on AVX-512
// without degrading AVX2 machines due to larger chunk size,
// the AVX-512 specialization should be re-enabled.
#[multiversion::multiversion(
    targets(
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
#[inline]
/// Separates data like `[1, 2, 3, 4]` into `([1, 3], [2, 4])` for any length
pub fn deinterleave_simd_swizzle_x86_64_v3<T: Copy + Default + SimdElement>(
    input: &[T],
) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;
    const DOUBLE_CHUNK: usize = CHUNK_SIZE * 2;

    let out_len = input.len() / 2;
    // We've benchmarked, and it turns out that this approach with zeroed memory
    // is faster than using uninit memory and bumping the length once in a while!
    let mut out_odd = vec![T::default(); out_len];
    let mut out_even = vec![T::default(); out_len];

    input
        .chunks_exact(DOUBLE_CHUNK)
        .zip(out_odd.chunks_exact_mut(CHUNK_SIZE))
        .zip(out_even.chunks_exact_mut(CHUNK_SIZE))
        .for_each(|((in_chunk, odds), evens)| {
            let in_simd: Simd<T, DOUBLE_CHUNK> = Simd::from_array(in_chunk.try_into().unwrap());
            // This generates *slightly* faster code than just assigning values by index.
            // You'd think simd::deinterleave would be appropriate, but it does something different!
            let result = simd_swizzle!(in_simd, [0, 2, 4, 6, 1, 3, 5, 7]);
            let result_arr = result.to_array();
            odds.copy_from_slice(&result_arr[..CHUNK_SIZE]);
            evens.copy_from_slice(&result_arr[CHUNK_SIZE..]);
        });

    // Process the remainder, too small for the vectorized loop
    let input_rem = input.chunks_exact(DOUBLE_CHUNK).remainder();
    let odds_rem = out_odd.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    let evens_rem = out_even.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    input_rem
        .chunks_exact(2)
        .zip(odds_rem.iter_mut())
        .zip(evens_rem.iter_mut())
        .for_each(|((inp, odd), even)| {
            *odd = inp[0];
            *even = inp[1];
        });

    (out_odd, out_even)
}

/// Slow but obviously correct implementation of deinterleaving,
/// to be used in tests and as a baselines for benchmarks
#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
    "x86_64+avx2+fma", // x86_64-v3
    "x86_64+sse4.2", // x86_64-v2
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    ))]
#[inline]
pub fn deinterleave_naive<T: Copy>(input: &[T]) -> (Vec<T>, Vec<T>) {
    input.chunks_exact(2).map(|c| (c[0], c[1])).unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(0, 0);
    }
}
