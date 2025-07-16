#![feature(portable_simd)]

use std::array;
use std::fmt::Debug;
use std::marker::PhantomData;

pub mod linalg;
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

macro_rules! HList {
		() => { Nil };
		($t:ty $(, $tn:ty $(: $rn:ty)?)*) => {
				Cons<$t, HList!($($tn $(: $rn)*),*)>
		};
		($t:ty : $r:ty $(, $tn:ty $(: $rn:ty)?)*) => {
				<Cons<$t, HList!($($tn $(: $rn)*),*)> as Repeat<$r>>::Output
		};
}

macro_rules! hlist {
		() => { Nil };
		($t:literal $(, $tn:literal)*) => {
				Cons($t, hlist!($($tn)*))
		};
}

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

trait ActivationFn<E> {
    fn activate<const S: usize>(v: &[E; S], i: usize) -> E;
}

#[derive(Default)]
struct Sigmoid;

impl<E: linalg::Number> ActivationFn<E> for Sigmoid {
    fn activate<const S: usize>(v: &[E; S], i: usize) -> E {
        E::one() / (E::get_exp(-v[i]) + E::one())
    }
}

#[derive(Default)]
struct Relu;

impl<E: linalg::Number> ActivationFn<E> for Relu {
    fn activate<const S: usize>(v: &[E; S], i: usize) -> E {
        E::get_max(v[i], E::default())
    }
}

trait FeedforwardTimes<Idx, Times> {
    fn feedforward_times(&mut self) -> &mut Self;
}

struct NeuralNetwork<Wba: ListTrio, F: List> {
    w: Wba::W,
    b: Wba::B,
    a: Wba::A,

    f: F,
}

impl<Idx, Wba: ListTrio, F: List> FeedforwardTimes<Idx, Zero> for NeuralNetwork<Wba, F> {
    fn feedforward_times(&mut self) -> &mut Self {
        self
    }
}

impl<Idx, Times, Wba, E, Fi, F, const AS: usize, const BS: usize, const CS: usize>
    FeedforwardTimes<Idx, Succ<Times>> for NeuralNetwork<Wba, F>
where
		Self: FeedforwardTimes<Succ<Idx>, Times>,
    Wba: ListTrio,
    Wba::W: Nth<Idx, Output = Layer<E, AS>>,
    Wba::B: Nth<Idx, Output = Layer<E, CS>>,
    Wba::A: Nth<Idx, Output = Layer<E, BS>> + Nth<Succ<Idx>, Output = Layer<E, CS>>,
    E: linalg::Number,
    Fi: ActivationFn<E>,
    F: List + Nth<Idx, Output = Fi>,
{
    fn feedforward_times(&mut self) -> &mut Self {
        let activations = &Nth::<Idx>::nth(&self.a).0;
        let weights = &Nth::<Idx>::nth(&self.w).0;
        let bias = &Nth::<Idx>::nth(&self.b).0;

        Nth::<Succ<Idx>>::nth_mut(&mut self.a).0 = {
            let activations = vvadd(&vmdot(activations, weights), bias);
            array::from_fn(|i| Fi::activate(&activations, i))
        };

        FeedforwardTimes::<Succ<Idx>, Times>::feedforward_times(self)
    }
}

impl<Wba: ListTrio, F: List> NeuralNetwork<Wba, F>
where
		Self: FeedforwardTimes<Zero, <Wba::W as Length>::Output>,
    Wba::W: Length,
{
    fn feedforward(&mut self) -> &mut Self {
        FeedforwardTimes::<Zero, <Wba::W as Length>::Output>::feedforward_times(self)
    }
}

trait ListTrio {
    type W: List;
    type B: List;
    type A: List;
}

impl<W, B, A> ListTrio for (W, B, A)
where
    W: List + Default,
    B: List + Default,
    A: List + Default,
{
    type W = W;
    type B = B;
    type A = A;
}

impl<Wba, F> NeuralNetwork<Wba, F>
where
    Wba: ListTrio,
    Wba::W: Default,
    Wba::B: Default,
    Wba::A: Default,
    F: List + Default,
{
    fn new() -> Self {
        Self {
            w: Wba::W::default(),
            b: Wba::B::default(),
            a: Wba::A::default(),
            f: F::default(),
        }
    }
}

macro_rules! Layers {
		(@wbuilder $t:ty; {$s:expr} $(: $r:ty)?) => {
				Nil
		};
		(@wbuilder $t:ty; {$sp:expr}, {$sn:expr} $(: $rn:ty)? $(, {$s:expr} $(: $r:ty)?)*) => {
				Cons<Layer<$t, {$sp*$sn}>, Layers!(@wbuilder $t; {$sn} $(: $rn)* $(, {$s} $(: $r)?)*)>
		};
		(@wbuilder $t:ty; {$sp:expr} : $rp:ty, {$sn:expr} $(: $rn:ty)? $(, {$s:expr} $(: $r:ty)?)*) => {
				<Cons<Layer<$t, {$sp*$sp}>,
				    Layers!(@wbuilder $t; {$sp}, {$sn} $(: $rn)* $(, {$s} $(: $r)?)*)
				> as Repeat<<$rp as Prec>::Output>>::Output
		};
		($t:ty; {$sn:expr} $(: $rn:ty)? $(, {$s:expr} $(: $r:ty)?)*) => {
				(
						Layers!(@wbuilder $t; {$sn} $(: $rn)* $(, {$s} $(: $r)*)*),
						HList!($(Layer<$t, $s> $(: $r)*),*),
						HList!(Layer<$t, $sn> $(: $rn)* $(, Layer<$t, $s> $(: $r)*)*),
				)
		};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_some_layers() {
        let (w, b, a) = <Layers!(f32; {7}, {4}: S2<S1>, {3})>::default();

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
        type Wba = Layers!(f32; {4}, {8}, {3});
        type F = HList!(Sigmoid: S2);

        let mut nn = NeuralNetwork::<Wba, F>::new();
        nn.w.0.0 = [1.; 32];
        nn.a.0.0 = [1.; 4];
        nn.w.1.0.0 = [1.; 24];
        nn.b.1.0.0 = [1.; 3];

        nn.feedforward();

        println!("{:?}", nn.a.0);
    }

    #[test]
    fn feedforward_sum() {
        type Wba = Layers!(f32; {2}, {1});
        type F = HList!(Relu);

        let mut nn = NeuralNetwork::<Wba, F>::new();
        nn.w.0.0 = [1., 1.];
        nn.a.0.0 = [2., 3.];

        nn.feedforward();
        assert!(f32::abs(nn.a.1.0.0[0] - 5.) < 0.000001);
    }
}
