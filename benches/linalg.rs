use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion};

use nerio::linalg::vmdot;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn get_system_counter() -> u64 {
    let c: u64;
    unsafe {
        std::arch::asm!("mrs {}, cntvct_el0", out(reg) c);
    }
    c
}

fn random_boxed_array<const S: usize>() -> Box<[f32; S]> {
    let mut v = Vec::with_capacity(S);

    for _ in 0..S {
        v.push(rand::random_range(-1.0..=1.0));
    }

    v.into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| panic!("unreachable"))
}

fn vmdot_bench<'a, const S: usize, const M: usize>(
    mut group: BenchmarkGroup<'a, WallTime>,
) -> BenchmarkGroup<'a, WallTime> {
    let v = random_boxed_array::<S>();
    let m = random_boxed_array::<M>();

    group.bench_function(format!("{}", S), |b| {
        b.iter(|| {
            let r: Box<[f32; S]> = Box::new(vmdot(black_box(&*v), black_box(&*m)));
            r
        })
    });
    group
}

fn criterion_benchmark(c: &mut Criterion) {
    let group = c.benchmark_group("vmdot");

    let group = vmdot_bench::<1024, { 1024 * 1024 }>(group);
    let group = vmdot_bench::<1536, { 1536 * 1536 }>(group);
    let group = vmdot_bench::<2048, { 2048 * 2048 }>(group);
    let group = vmdot_bench::<2560, { 2560 * 2560 }>(group);
    let group = vmdot_bench::<3072, { 3072 * 3072 }>(group);
    let group = vmdot_bench::<3584, { 3584 * 3584 }>(group);
    let _group = vmdot_bench::<4096, { 4096 * 4096 }>(group);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
