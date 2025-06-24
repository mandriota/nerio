#![feature(portable_simd)]

use std::array;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul};

mod linalg;
use linalg::{vmdot, vvadd};

fn sigmoid<E: linalg::Number>(x: E) -> E {
    E::one() / (E::get_exp(-x) + E::one())
}

fn sigmoid_derivative<E: linalg::Number>(x: E) -> E {
    x * (E::one() - x)
}

trait List {}

#[derive(Default, Debug, Clone)]
struct Nil;
#[derive(Default, Debug, Clone)]
struct Cons<H, T>(H, T);

impl List for Nil {}
impl<H, T: List> List for Cons<H, T> {}

#[derive(Default)]
struct Zero;
#[derive(Default)]
struct Succ<N>(PhantomData<N>);

type S1<N = Zero> = Succ<N>;
type S2<N = Zero> = S1<S1<N>>;
type S4<N = Zero> = S2<S2<N>>;
type S8<N = Zero> = S4<S4<N>>;
type S16<N = Zero> = S8<S8<N>>;
type S32<N = Zero> = S16<S16<N>>;
type S64<N = Zero> = S32<S32<N>>;

trait Prec {
    type Output;
}

impl<N> Prec for Succ<N> {
    type Output = N;
}

trait Repeat<N> {
    type Output;
}

impl<H, T: List> Repeat<Zero> for Cons<H, T> {
    type Output = T;
}

impl<H, T: List, N> Repeat<Succ<N>> for Cons<H, T>
where
    Cons<H, T>: Repeat<N>,
{
    type Output = Cons<H, <Cons<H, T> as Repeat<N>>::Output>;
}

trait Length {
    type Output;
}

impl Length for Nil {
    type Output = Zero;
}

impl<H, T: Length> Length for Cons<H, T> {
    type Output = Succ<<T as Length>::Output>;
}

trait Nth<Idx> {
    type Output;

    fn nth(&self) -> &Self::Output;
    fn nth_mut(&mut self) -> &mut Self::Output;
}

impl<H, T> Nth<Zero> for Cons<H, T> {
    type Output = H;

    fn nth(&self) -> &Self::Output {
        &self.0
    }

    fn nth_mut(&mut self) -> &mut Self::Output {
        &mut self.0
    }
}

impl<Idx, H, T> Nth<Succ<Idx>> for Cons<H, T>
where
    T: Nth<Idx>,
{
    type Output = T::Output;

    fn nth(&self) -> &Self::Output {
        Nth::<Idx>::nth(&self.1)
    }

    fn nth_mut(&mut self) -> &mut Self::Output {
        Nth::<Idx>::nth_mut(&mut self.1)
    }
}

#[derive(Debug)]
struct Layer<E, const S: usize>([E; S]);

impl<E: Default, const S: usize> Default for Layer<E, S> {
		fn default() -> Self {
				Layer(array::from_fn(|_| E::default()))
		}
}

trait Builder: Sized {
    fn before<F>(self) -> Cons<F, Self>
    where
        F: Default,
    {
        Cons(F::default(), self)
    }

    fn repeat<N>(self) -> <Self as Repeat<N>>::Output
    where
        Self: List + Repeat<N>,
        <Self as Repeat<N>>::Output: Default,
    {
        <Self as Repeat<N>>::Output::default()
    }
}

impl Builder for Nil {}
impl<H, T> Builder for Cons<H, T> {}

trait ActivationFn<E> {
		fn activate(e: E) -> E;
}

#[derive(Default)]
struct Sigmoid;

impl<E: linalg::Number> ActivationFn<E> for Sigmoid {
		fn activate(x: E) -> E {
				E::one() / (E::get_exp(-x) + E::one())
		}
}

#[derive(Default)]
struct Relu;

impl<E: linalg::Number> ActivationFn<E> for Relu {
		fn activate(x: E) -> E {
				E::get_max(x, E::default())
		}
}

trait FeedforwardTimes<Idx, Times> {
    fn feedforward_times(&mut self) -> &mut Self;
}

struct NeuralNetwork<W: List, B: List, A: List, F: List> {
    w: W,
    b: B,
    a: A,
		
		f: F,
}

impl<Idx, W: List, B: List, A: List, F: List> FeedforwardTimes<Idx, Zero> for NeuralNetwork<W, B, A, F> {
    fn feedforward_times(&mut self) -> &mut Self {
        self
    }
}

impl<Idx, Times, W, B, A, E, Fi, F, const AS: usize, const BS: usize, const CS: usize>
    FeedforwardTimes<Idx, Succ<Times>> for NeuralNetwork<W, B, A, F>
where
    E: linalg::Number,
		Fi: ActivationFn<E>,
    F: List + Nth<Idx, Output = Fi>,
    W: List + Nth<Idx, Output = Layer<E, AS>>,
    B: List + Nth<Idx, Output = Layer<E, CS>>,
    A: List + Nth<Idx, Output = Layer<E, BS>> + Nth<Succ<Idx>, Output = Layer<E, CS>>,
    Self: FeedforwardTimes<Succ<Idx>, Times>,
{
    fn feedforward_times(&mut self) -> &mut Self {
				let activate = Nth::<Idx>::nth(&self.f);
        let activations = &Nth::<Idx>::nth(&self.a).0;
        let weights = &Nth::<Idx>::nth(&self.w).0;
        let bias = &Nth::<Idx>::nth(&self.b).0;

        let activations = vvadd(&vmdot(activations, weights), bias).map(|e| Fi::activate(e));
        Nth::<Succ<Idx>>::nth_mut(&mut self.a).0 = activations;

        FeedforwardTimes::<Succ<Idx>, Times>::feedforward_times(self)
    }
}

impl<W: List + Length, B: List, A: List, F: List> NeuralNetwork<W, B, A, F>
where
    Self: FeedforwardTimes<Zero, <W as Length>::Output>,
{
    fn feedforward(&mut self) -> &mut Self {
        FeedforwardTimes::<Zero, <W as Length>::Output>::feedforward_times(self)
    }
}

impl<W: List, B: List, A: List, F: List> NeuralNetwork<W, B, A, F>
{
		fn new(wba: (W, B, A), f: F) -> Self {
				Self {
						w: wba.0,
						b: wba.1,
						a: wba.2,
						f,
				}
		}
}

macro_rules! layers {
		(@wbuilder $rt:expr; $t:ty; {$s:expr} $(x $r:ty)?) => {
				$rt
		};
		(@wbuilder $rt:expr; $t:ty; {$sp:expr} $(x $rp:ty)?, {$sn:expr} $(x $rn:ty)? $(, {$s:expr} $(x $r:ty)?)*) => {
				layers!(
						@wbuilder
						$rt
						    $(
							      .before::<Layer<$t, {$sp*$sp}>>()
							      .repeat::<<$rp as Prec>::Output>()
							  )*
							  .before::<Layer<$t, {$sp*$sn}>>();
						$t;
						{$sn} $(x $rn)* $(, {$s} $(x $r)*)*
				)
		};
		(@bbuilder $rt:expr; $t:ty; {$s:expr} $(x $r:ty)?) => {
				$rt
						$(
								.before::<Layer<$t, $s>>()
								.repeat::<<$r as Prec>::Output>()
						)*
		};
		(@bbuilder $rt:expr; $t:ty; {$sp:expr} $(x $rp:ty)? $(, {$s:expr} $(x $r:ty)?)*) => {
				layers!(
						@bbuilder
						$rt
						    .before::<Layer<$t, $sp>>()
							  $(.repeat::<$rp>())*;
						$t;
						$({$s} $(x $r)*),*
				)
		};
		($t:ty; $({$s:expr} $(x $r:ty)?),+) => {
				(
						layers!(@wbuilder Nil; $t; $({$s} $(x $r)*),*),
						layers!(@bbuilder Nil; $t; $({$s} $(x $r)*),*),
						Nil
								$(
										.before::<Layer<$t, $s>>()
										$(.repeat::<$r>())*
								)*,
				)
		};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_some_layers() {
        // currently layers are set in reverse order
        let (w, b, a) = layers!(f32; {3}, {4} x S2<S1>, {7});

				assert_eq!(w.0.0.len(), 28);
        assert_eq!(w.1.0.0.len(), 16);
        assert_eq!(w.1.1.0.0.len(), 16);
        assert_eq!(w.1.1.1.0.0.len(), 12);

				assert_eq!(b.0.0.len(), 4);
        assert_eq!(b.1.0.0.len(), 4);
        assert_eq!(b.1.1.0.0.len(), 4);
        assert_eq!(b.1.1.1.0.0.len(), 3);

				assert_eq!(a.0.0.len(), 7);
        assert_eq!(a.1.0.0.len(), 4);
        assert_eq!(a.1.1.0.0.len(), 4);
        assert_eq!(a.1.1.1.0.0.len(), 4);
        assert_eq!(a.1.1.1.1.0.0.len(), 3);
    }
		
    #[test]
    fn some_feedforward() {
        let wba = layers!(f32; {3}, {8}, {4});
				let f = Nil
						.before::<Sigmoid>()
						.repeat::<S2>();

        let mut nn = NeuralNetwork::new(wba, f);
				nn.w.0.0 = [1.; 32];
        nn.a.0.0 = [1.; 4];
        nn.w.1.0.0 = [1.; 24];
        nn.b.1.0.0 = [1.; 3];

        nn.feedforward();

        println!("{:?}", nn.a.0);
    }

		#[test]
		fn feedforward_sum() {
				let wba = layers!(f32; {1}, {2});
				let f = Nil.before::<Relu>();

				let mut nn = NeuralNetwork::new(wba, f);
				nn.w.0.0 = [1., 1.];
				nn.a.0.0 = [2., 3.];

				nn.feedforward();
				assert!(f32::abs(nn.a.1.0.0[0] - 5.) < 0.000001);
		}
}
