#![forbid(unsafe_code)]
#![feature(portable_simd, avx512_target_feature)]

use std::simd::{simd_swizzle, Simd, SimdElement};

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
pub fn deinterleave_simd_unpck_x86_64_v4<T: Copy + Default + SimdElement>(
    input: &[T],
) -> (Vec<T>, Vec<T>) {
    const CHUNK_SIZE: usize = 4;
    const DOUBLE_CHUNK: usize = CHUNK_SIZE * 2;
    let out_len = input.len() / 2;
    let mut reals = vec![T::default(); out_len];
    let mut imags = vec![T::default(); out_len];

    for ((chunk, chunk_re), chunk_im) in input
        .chunks_exact(DOUBLE_CHUNK)
        .zip(reals.chunks_exact_mut(CHUNK_SIZE))
        .zip(imags.chunks_exact_mut(CHUNK_SIZE))
    {
        let (first_half, second_half) = chunk.split_at(CHUNK_SIZE);

        let a = Simd::<T, CHUNK_SIZE>::from_slice(first_half);
        let b = Simd::<T, CHUNK_SIZE>::from_slice(second_half);
        let (re_deinterleaved, im_deinterleaved) = a.deinterleave(b);

        chunk_re.copy_from_slice(&re_deinterleaved.to_array());
        chunk_im.copy_from_slice(&im_deinterleaved.to_array());
    }

    let remainder = input.chunks_exact(DOUBLE_CHUNK).remainder();
    let reals_rem = reals.chunks_exact_mut(CHUNK_SIZE).into_remainder();
    let imags_rem = imags.chunks_exact_mut(CHUNK_SIZE).into_remainder();

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
    fn unpack_high_low_deinterleave() {
        for i in 0..20 {
            let n = 1 << i;
            let mut interleaved_vec = vec![0.0; n * 2];

            interleaved_vec.chunks_exact_mut(2).for_each(|c| {
                c[0] = 1.0;
                c[1] = 0.0
            });
            println!("{interleaved_vec:?}");

            let (e, o) = deinterleave_simd_unpck_x86_64_v4(&interleaved_vec);

            println!("{e:?}\n{o:?}");

            assert_eq!(e, vec![1.0; n]);
            assert_eq!(o, vec![0.0; n]);
        }
    }
}
