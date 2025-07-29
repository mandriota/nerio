use std::array;
use std::marker::Copy;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::simd::{f32x4, simd_swizzle};

pub trait Accum: Sized + AddAssign {}
impl<T> Accum for T where T: AddAssign {}

pub trait RepeatableAccum: Accum + Mul<Output = Self> {}
impl<T> RepeatableAccum for T where T: Accum + Mul<Output = Self> {}

pub trait Number:
    RepeatableAccum
    + Default
    + Copy
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
		+ for<'a> std::iter::Sum<&'a Self>
{
    fn get_exp(self) -> Self;
    fn get_max(self, r: Self) -> Self;

    fn one() -> Self {
        Self::get_exp(Self::default())
    }
}

impl Number for f32 {
    fn get_exp(self) -> Self {
        f32::exp(self)
    }

    fn get_max(self, r: Self) -> Self {
        f32::max(self, r)
    }
}

impl Number for f64 {
    fn get_exp(self) -> Self {
        f64::exp(self)
    }

    fn get_max(self, r: Self) -> Self {
        f64::max(self, r)
    }
}

fn mat_block2x2_dot(a: f32x4, b: f32x4) -> f32x4 {
    let adad = simd_swizzle!(a, [0, 3, 0, 3]);
    let cbcb = simd_swizzle!(a, [2, 1, 2, 1]);
    let egfh = simd_swizzle!(b, [0, 2, 1, 3]);

    let prod_x = adad * egfh;
    let prod_y = cbcb * egfh;
    let sum = prod_x + simd_swizzle!(prod_y, [1, 0, 3, 2]);
    simd_swizzle!(sum, [0, 2, 1, 3])
}

pub fn vvadd<T, const N: usize>(a: &[T; N], b: &[T; N]) -> [T; N]
where
    T: Copy + Add<Output = T>,
{
    array::from_fn(|i| a[i] + b[i])
}

pub fn vvhadamard<T, const N: usize>(a: &[T; N], b: &[T; N]) -> [T; N]
where
    T: Copy + Mul<Output = T>,
{
    array::from_fn(|i| a[i] * b[i])
}

// TODO: try block based vector by matrix dot product
pub fn vmdot<T, const VS: usize, const MC: usize>(v: &[T; VS], m: &[[T; VS]; MC]) -> [T; MC]
where
    T: Copy + RepeatableAccum,
{
    // TODO: compare assembly with normal iteration (without flattening)
    let m_flattened = m.as_flattened();

    array::from_fn(|i| {
        let mut rt = m_flattened[i * VS] * v[0];

        for j in 1..VS {
            rt += m_flattened[i * VS + j] * v[j];
        }

        rt
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_by_vector_dot_product() {
        let v = [1., 2., 1.];
        let m = [[2., 1., 7.], [6., 5., 8.]];

        let r: [f32; 2] = vmdot(&v, &m);

        assert!(f32::abs(r[0] - 11.) < 0.000001);
        assert!(f32::abs(r[1] - 24.) < 0.000001);
    }
}
