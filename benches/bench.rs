#![feature(portable_simd, avx512_target_feature)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use deinterleave_rs::{
    deinterleave_autovec, deinterleave_naive, deinterleave_simd_swizzle_x86_64_v3,
    deinterleave_simd_swizzle_x86_64_v4,
};

fn benchmark_deinterleave(c: &mut Criterion) {
    let mut group = c.benchmark_group("deinterleave");

    for s in (4..=28).step_by(4) {
        let size = 1 << s;
        let input: Vec<f64> = (0..size).map(|x| x as f64).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("Naive deinterleave", size),
            &input,
            |b, input| b.iter(|| deinterleave_naive(black_box(input))),
        );

        group.bench_with_input(
            BenchmarkId::new("Autovectorized deinterleave", size),
            &input,
            |b, input| b.iter(|| deinterleave_autovec(black_box(input))),
        );

        group.bench_with_input(
            BenchmarkId::new("Simd Swizzle deinterleave x84-64-v3", size),
            &input,
            |b, input| b.iter(|| deinterleave_simd_swizzle_x86_64_v3(black_box(input))),
        );

        group.bench_with_input(
            BenchmarkId::new("Simd Swizzle deinterleave x84-64-v4", size),
            &input,
            |b, input| b.iter(|| deinterleave_simd_swizzle_x86_64_v4(black_box(input))),
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_deinterleave);
criterion_main!(benches);
