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

trait ListQuartet {
    type W: List;
    type B: List;
    type Z: List;
    type A: List;
}

impl<W, B, Z, A> ListQuartet for (W, B, Z, A)
where
    W: List + Default,
    B: List + Default,
    Z: List + Default,
    A: List + Default,
{
    type W = W;
    type B = B;
    type Z = Z;
    type A = A;
}

#[derive(Debug)]
struct Layer<E, const R: usize, const C: usize>([[E; R]; C]);

impl<E: Default, const R: usize, const C: usize> Default for Layer<E, R, C> {
    fn default() -> Self {
        Layer(array::from_fn(|_| array::from_fn(|_| E::default())))
    }
}

trait ActivationFn<E> {
    fn activate<const S: usize>(z: &[E; S], i: usize) -> E;
}

#[derive(Default)]
struct Sigmoid;

impl<E: linalg::Number> ActivationFn<E> for Sigmoid {
    fn activate<const S: usize>(z: &[E; S], i: usize) -> E {
        E::one() / (E::get_exp(-z[i]) + E::one())
    }
}

#[derive(Default)]
struct Relu;

impl<E: linalg::Number> ActivationFn<E> for Relu {
    fn activate<const S: usize>(z: &[E; S], i: usize) -> E {
        E::get_max(z[i], E::default())
    }
}

trait FeedforwardAt<Idx, RemLayers, E, const BS: usize> {
    type OutputLayer;

    fn feedforward_at(&self, a: Layer<E, BS, 1>) -> Self::OutputLayer;
}

struct Feedforwarder<W: List, B: List, F: List> {
    w: W,
    b: B,

    f: F,
}

impl<Idx, W: List, B: List, E, F: List, const FINAL: usize> FeedforwardAt<Idx, Zero, E, FINAL>
    for Feedforwarder<W, B, F>
{
    type OutputLayer = Layer<E, FINAL, 1>;

    fn feedforward_at(&self, a: Self::OutputLayer) -> Self::OutputLayer {
        a
    }
}

impl<Idx, RemLayers, W, B, E, Fi, F, const FEED: usize, const SINK: usize, const FINAL: usize>
    FeedforwardAt<Idx, Succ<RemLayers>, E, FEED> for Feedforwarder<W, B, F>
where
    Self: FeedforwardAt<Succ<Idx>, RemLayers, E, SINK, OutputLayer = Layer<E, FINAL, 1>>,
    W: List + Nth<Idx, Output = Layer<E, FEED, SINK>>,
    B: List + Nth<Idx, Output = Layer<E, SINK, 1>>,
    E: linalg::Number,
    Fi: ActivationFn<E>,
    F: List + Nth<Idx, Output = Fi>,
{
    type OutputLayer = Layer<E, FINAL, 1>;

    fn feedforward_at(&self, a: Layer<E, FEED, 1>) -> Self::OutputLayer {
        let weights = &Nth::<Idx>::nth(&self.w).0;
        let bias = &Nth::<Idx>::nth(&self.b).0[0];

        let result: [E; SINK] =
            array::from_fn(|i| Fi::activate(&vvadd(&vmdot(&a.0[0], weights), bias), i));

        FeedforwardAt::<Succ<Idx>, RemLayers, E, SINK>::feedforward_at(self, Layer([result]))
    }
}

impl<W, B, E, Fi, F, const FEED: usize, const SINK: usize, const FINAL: usize>
    Feedforwarder<W, B, F>
where
    Self: FeedforwardAt<Zero, <B as Length>::Output, E, FEED, OutputLayer = Layer<E, FINAL, 1>>,
    W: List + Nth<Zero, Output = Layer<E, FEED, SINK>>,
    B: List + Nth<Zero, Output = Layer<E, SINK, 1>> + Length,
    E: linalg::Number,
    Fi: ActivationFn<E>,
    F: List + Nth<Zero, Output = Fi>,
{
    fn feedforward(&self, a: Layer<E, FEED, 1>) -> Layer<E, FINAL, 1> {
        FeedforwardAt::<Zero, <B as Length>::Output, E, FEED>::feedforward_at(self, a)
    }
}

trait TracingFeedforwardAt<Idx, RemLayers> {
    fn tracing_feedforward_at(&mut self) -> &mut Self;
}

// DONE: make separate structure FeedforwarderNetwork for prediction only
// Or maybe name it Predictor? Anyway...
struct NeuralNetwork<Wbza: ListQuartet, F: List> {
    f: Feedforwarder<Wbza::W, Wbza::B, F>,

    z: Wbza::Z,
    a: Wbza::A,
}

impl<Idx, Wbza: ListQuartet, F: List> TracingFeedforwardAt<Idx, Zero> for NeuralNetwork<Wbza, F> {
    fn tracing_feedforward_at(&mut self) -> &mut Self {
        self
    }
}

// TODO: maybe add separate function which just does feedforward without modifying NeuralNetwork data
// It should allocate array with size of largest layers - then just use transmute for each layer.
// Array should be passed as argument of the function.

impl<Idx, RemLayers, Wbza, E, Fi, F, const FEED: usize, const SINK: usize>
    TracingFeedforwardAt<Idx, Succ<RemLayers>> for NeuralNetwork<Wbza, F>
where
    Self: TracingFeedforwardAt<Succ<Idx>, RemLayers>,
    Wbza: ListQuartet,
    Wbza::W: Nth<Idx, Output = Layer<E, FEED, SINK>>,
    Wbza::B: Nth<Idx, Output = Layer<E, SINK, 1>>,
    Wbza::Z: Nth<Idx, Output = Layer<E, FEED, 1>> + Nth<Succ<Idx>, Output = Layer<E, SINK, 1>>,
    Wbza::A: Nth<Idx, Output = Layer<E, FEED, 1>> + Nth<Succ<Idx>, Output = Layer<E, SINK, 1>>,
    E: linalg::Number,
    Fi: ActivationFn<E>,
    F: List + Nth<Idx, Output = Fi>,
{
    fn tracing_feedforward_at(&mut self) -> &mut Self {
        // TODO: add switcher (in case switcher is not active, do not store activations even in array)
        let activations = &Nth::<Idx>::nth(&self.a).0[0];
        let weights = &Nth::<Idx>::nth(&self.f.w).0;
        let bias = &Nth::<Idx>::nth(&self.f.b).0[0];

        Nth::<Succ<Idx>>::nth_mut(&mut self.z).0[0] = vvadd(&vmdot(activations, weights), bias);
        Nth::<Succ<Idx>>::nth_mut(&mut self.a).0[0] =
            array::from_fn(|i| Fi::activate(&Nth::<Succ<Idx>>::nth(&self.z).0[0], i));

        TracingFeedforwardAt::<Succ<Idx>, RemLayers>::tracing_feedforward_at(self)
    }
}

trait BackpropAt<Idx> {
    type OutputLayer;

    fn backprop_at(&mut self, expected: Self::OutputLayer) -> &mut Self;
}

impl<Wbza: ListQuartet, F: List> BackpropAt<Zero> for NeuralNetwork<Wbza, F>
where
    Wbza::A: Nth<Zero>,
{
    type OutputLayer = <Wbza::A as Nth<Zero>>::Output;

    fn backprop_at(&mut self, _expected: Self::OutputLayer) -> &mut Self {
        self
    }
}

impl<Idx, Wbza: ListQuartet, E, F: List, const OS: usize> BackpropAt<Succ<Idx>>
    for NeuralNetwork<Wbza, F>
where
    Self: BackpropAt<Idx>,
    E: Default, // TODO: remove (currently it is used for test)
    <Self as BackpropAt<Idx>>::OutputLayer: Default, // TODO: remove
    Wbza::A: Nth<Idx, Output = <Self as BackpropAt<Idx>>::OutputLayer>
        + Nth<Succ<Idx>, Output = Layer<E, OS, 1>>,
{
    type OutputLayer = <Wbza::A as Nth<Succ<Idx>>>::Output;

    fn backprop_at(&mut self, _expected: Self::OutputLayer) -> &mut Self {
        println!("backprop iteration");
        BackpropAt::<Idx>::backprop_at(self, <Wbza::A as Nth<Idx>>::Output::default())
    }
}

// TODO: use mostly B as reference length, because it is equal to W length.
// In this way, in theory, Length trait may be cached by the prover.
impl<Wbza: ListQuartet, F: List> NeuralNetwork<Wbza, F>
where
    Self: TracingFeedforwardAt<Zero, <Wbza::W as Length>::Output>
        + BackpropAt<<Wbza::W as Length>::Output>,
    Wbza::W: Length,
{
    // TODO: maybe remove wrapper when non tracing feedforward will be implemented.
    // Instead, normally, user should use non tracing feedforward.
    fn tracing_feedforward(&mut self) -> &mut Self {
        TracingFeedforwardAt::<Zero, <Wbza::W as Length>::Output>::tracing_feedforward_at(self)
    }

    fn backprop(
        &mut self,
        expected: <Self as BackpropAt<<Wbza::W as Length>::Output>>::OutputLayer,
    ) -> &mut Self {
        self.tracing_feedforward(); // TODO: add "switcher" to FeedforwardTimes trait, marking if to store z separately from a
        BackpropAt::<<Wbza::W as Length>::Output>::backprop_at(self, expected)
    }
}

impl<Wbza, F> NeuralNetwork<Wbza, F>
where
    Wbza: ListQuartet,
    Wbza::W: Default, // TODO: don't use Default, instead initialize weights and biases randomly \
    Wbza::B: Default, // considering type of activation function
    Wbza::Z: Default,
    Wbza::A: Default,
    F: List + Default,
{
    fn new() -> Self {
        Self {
            f: Feedforwarder {
                w: Wbza::W::default(),
                b: Wbza::B::default(),
                f: F::default(),
            },
            z: Wbza::Z::default(),
            a: Wbza::A::default(),
        }
    }
}

macro_rules! Layers {
		(@wbuilder $t:ty; {$s:expr} $(: $r:ty)?) => {
				Nil
		};
		(@wbuilder $t:ty; {$sp:expr}, {$sn:expr} $(: $rn:ty)? $(, {$s:expr} $(: $r:ty)?)*) => {
				Cons<Layer<$t, $sp, $sn>, Layers!(@wbuilder $t; {$sn} $(: $rn)* $(, {$s} $(: $r)?)*)>
		};
		(@wbuilder $t:ty; {$sp:expr} : $rp:ty, {$sn:expr} $(: $rn:ty)? $(, {$s:expr} $(: $r:ty)?)*) => {
				<Cons<Layer<$t, $sp, $sp>,
				    Layers!(@wbuilder $t; {$sp}, {$sn} $(: $rn)* $(, {$s} $(: $r)?)*)
				> as Repeat<<$rp as Prec>::Output>>::Output
		};
		($t:ty; {$sn:expr} $(: $rn:ty)? $(, {$s:expr} $(: $r:ty)?)*) => {
				(
						Layers!(@wbuilder $t; {$sn} $(: $rn)* $(, {$s} $(: $r)*)*),
						HList!($(Layer<$t, $s, 1> $(: $r)*),*),
						HList!(Layer<$t, $sn, 1> $(: $rn)* $(, Layer<$t, $s, 1> $(: $r)*)*),
						HList!(Layer<$t, $sn, 1> $(: $rn)* $(, Layer<$t, $s, 1> $(: $r)*)*),
				)
		};
}

pub fn feedforward_sum() {
    type Wbza = Layers!(f32; {2}, {9}: S8, {1});
    type F = HList!(Relu: S8<S1>);

    let mut nn = NeuralNetwork::<Wbza, F>::new();
    nn.f.w.0.0 = [[1.; 2]; 9];
    nn.a.0.0 = [[2., 3.]; 1];

    nn.tracing_feedforward();
    assert!(f32::abs(nn.a.1.0.0[0][0] - 5.) < 0.000001);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_some_layers() {
        let (w, b, z, a) = <Layers!(f32; {7}, {4}: S2<S1>, {3})>::default();

        assert_eq!(w.0.0.len(), 4);
        assert_eq!(w.0.0[0].len(), 7);
        assert_eq!(w.1.0.0.len(), 4);
        assert_eq!(w.1.0.0[0].len(), 4);
        assert_eq!(w.1.1.0.0.len(), 4);
        assert_eq!(w.1.0.0[0].len(), 4);
        assert_eq!(w.1.1.1.0.0.len(), 3);
        assert_eq!(w.1.1.1.0.0[0].len(), 4);

        assert_eq!(b.0.0.len(), 1);
        assert_eq!(b.0.0[0].len(), 4);
        assert_eq!(b.1.0.0.len(), 1);
        assert_eq!(b.1.0.0[0].len(), 4);
        assert_eq!(b.1.1.0.0.len(), 1);
        assert_eq!(b.1.1.0.0[0].len(), 4);
        assert_eq!(b.1.1.1.0.0.len(), 1);
        assert_eq!(b.1.1.1.0.0[0].len(), 3);

        assert_eq!(z.0.0.len(), 1);
        assert_eq!(z.0.0[0].len(), 7);
        assert_eq!(z.1.0.0.len(), 1);
        assert_eq!(z.1.0.0[0].len(), 4);
        assert_eq!(z.1.1.0.0.len(), 1);
        assert_eq!(z.1.1.0.0[0].len(), 4);
        assert_eq!(z.1.1.1.0.0.len(), 1);
        assert_eq!(z.1.1.1.0.0[0].len(), 4);
        assert_eq!(z.1.1.1.1.0.0.len(), 1);
        assert_eq!(z.1.1.1.1.0.0[0].len(), 3);

        assert_eq!(a.0.0.len(), 1);
        assert_eq!(a.0.0[0].len(), 7);
        assert_eq!(a.1.0.0.len(), 1);
        assert_eq!(a.1.0.0[0].len(), 4);
        assert_eq!(a.1.1.0.0.len(), 1);
        assert_eq!(a.1.1.0.0[0].len(), 4);
        assert_eq!(a.1.1.1.0.0.len(), 1);
        assert_eq!(a.1.1.1.0.0[0].len(), 4);
        assert_eq!(a.1.1.1.1.0.0.len(), 1);
        assert_eq!(a.1.1.1.1.0.0[0].len(), 3);
    }

    #[test]
    fn some_tracing_feedforward() {
        type Wbza = Layers!(f32; {4}, {8}, {3});
        // TODO: add troubleshooting section to docs about what obscure rust errors may indicate
        // Including this ("... but its trait bounds were not satisfied"):
        // It often indicated that Length of F is shorter than L-1.
        type F = HList!(Sigmoid: S2);

        let mut nn = NeuralNetwork::<Wbza, F>::new();
        nn.f.w.0.0 = [[1.; 4]; 8];
        nn.a.0.0 = [[1.; 4]];
        nn.f.w.1.0.0 = [[1.; 8]; 3];
        nn.f.b.1.0.0 = [[1.; 3]];

        nn.tracing_feedforward();

        println!("{:?}", nn.a.0);
    }

    #[test]
    fn tracing_feedforward_sum() {
        type Wbza = Layers!(f32; {2}, {1});
        type F = HList!(Relu);

        let mut nn = NeuralNetwork::<Wbza, F>::new();
        nn.f.w.0.0 = [[1., 1.]];
        nn.a.0.0 = [[2., 3.]];

        nn.tracing_feedforward();
        assert!(f32::abs(nn.a.1.0.0[0][0] - 5.) < 0.000001);
    }

    #[test]
    fn feedforward_triple_sum() {
        type Wbza = Layers!(f32; {2}, {3}, {1});
        type F = HList!(Relu: S2);

        let mut nn = NeuralNetwork::<Wbza, F>::new();
        nn.f.w.0.0 = [[1.; 2]; 3];
        nn.f.w.1.0.0 = [[1.; 3]];

        let result = nn.f.feedforward(Layer([[2., 5.]]));

        // NOTE: Network evaluation should be (2+5)*3=21
        assert!(f32::abs(result.0[0][0] - 21.) < 0.000001);
    }

    #[test]
    fn some_backprop() {
        type Wbza = Layers!(f32; {4}, {8}, {3});
        type F = HList!(Sigmoid: S2);

        let mut nn = NeuralNetwork::<Wbza, F>::new();

        nn.backprop(Layer([[0.; 3]]));
    }
}
