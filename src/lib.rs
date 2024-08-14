#![forbid(unsafe_code)]
#![feature(portable_simd, avx512_target_feature)]

use bytemuck::cast_slice;
use num_complex::Complex;
// use num_traits::Float;
use std::simd::{f32x16, f64x8};

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
            if !remainder.is_empty() {
                let i = reals.len() - remainder.len() / 2;
            remainder
                .chunks_exact(2)
                .zip(reals[i..].iter_mut())
                .zip(imags[i..].iter_mut())
                .for_each(|((c, re), im)| {
                    *re = c[0];
                    *im = c[1];
                });
            }

            (reals, imags)
        }
    };
}

impl_separate_re_im!(simd_deinterleave_32, f32, 16, f32x16);
impl_separate_re_im!(simd_deinterleave_64, f64, 8, f64x8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(0, 0);
    }
}
