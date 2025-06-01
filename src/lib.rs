#![recursion_limit = "1024"]

use std::marker::PhantomData;

fn sigmoid(x: f32) -> f32 {
    1. / (f32::exp(-x) + 1.)
}

fn sigmoid_derivative(x: f32) -> f32 {
    x * (1. - x)
}

trait List {}

#[derive(Default)]
struct Nil;
#[derive(Default)]
struct Cons<H, T>(H, T);

impl List for Nil {}
impl<H, T: List> List for Cons<H, T> {}

struct Zero;
struct Succ<N>(PhantomData<N>);

type Succ2<N> = Succ<Succ<N>>;
type Succ4<N> = Succ2<Succ2<N>>;
type Succ8<N> = Succ4<Succ4<N>>;
type Succ16<N> = Succ8<Succ8<N>>;
type Succ32<N> = Succ16<Succ16<N>>;
type Succ64<N> = Succ32<Succ32<N>>;

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

trait Nth<N> {
    type Output;
}

impl<H, T> Nth<Zero> for Cons<H, T> {
    type Output = H;
}

impl<H, T, N> Nth<Succ<N>> for Cons<H, T>
where
    T: Nth<N>,
{
    type Output = T::Output;
}

trait Builder: Sized {
    fn then<F>(self) -> Cons<F, Self>
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

struct NeuralNetwork<LWB: List>
where
    LWB: Nth<Zero> + Nth<Succ<Zero>> + Nth<Succ<Succ<Zero>>>,
{
    l: <LWB as Nth<Zero>>::Output,
    w: <LWB as Nth<Succ<Zero>>>::Output,
    b: <LWB as Nth<Succ2<Zero>>>::Output,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_some_layers() {
        // currently layers are set in reverse order
        let r = Nil
            .then::<[f32; 3]>()
            .then::<[f32; 12]>()
            .repeat::<Succ2<Succ<Zero>>>()
            .then::<[f32; 7]>();

        assert_eq!(r.0.len(), 7);
        assert_eq!(r.1.0.len(), 12);
        assert_eq!(r.1.1.0.len(), 12);
        assert_eq!(r.1.1.1.0.len(), 12);
        assert_eq!(r.1.1.1.1.0.len(), 3);
    }
}
