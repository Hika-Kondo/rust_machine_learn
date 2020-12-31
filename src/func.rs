use ndarray::{Dimension, Array};
use ndarray_linalg::types::Scalar;

// Exp for ndarray
pub trait Exp {
    type ExpType;
    fn exp(&self) -> Self::ExpType;
}

impl<T: Scalar,  U: Dimension> Exp for Array<T, U> {
    type ExpType = Array<T, U>;
    fn exp(& self) -> Self::ExpType { 
        self.map(|x| x.exp())
        // self.exp()
    }
}

// Sigmoid for Float and Array
pub trait Sigmoid {
    type SigmoidType;
    fn sigmoid(&self) -> Self::SigmoidType;
}

impl<F: Scalar, U: Dimension > Sigmoid for Array<F, U> {
    type SigmoidType = Array<F, U>;
    fn sigmoid(&self) -> Self::SigmoidType {
        // let one: T = num_traits::cast(1).unwrap();
        // let ni: T = num_traits::cast(-1).unwrap();
        // self.map(|x| one / (one + (*x * ni).exp()))
        let ones = Array::<F, U>::ones(self.raw_dim());
        let mimus = Array::<F, U>::zeros(self.raw_dim()) - self;
        ones.clone() / (ones + &mimus.exp())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use ndarray::*;
    use approx::abs_diff_eq;
    #[test]
    fn test_exp() {
        // for Array2<f64>
        let mut input: Array2<f64> = arr2(&[[1f64, 1f64], [1f64, 1f64]]);
        input = input.exp();
        let target: Array2<f64> = arr2(&[
            [2.718281828459045, 2.718281828459045],
            [2.718281828459045, 2.718281828459045]]);
        abs_diff_eq!(input, target, epsilon=1e-12);

        // for Array1<f32>
        let mut input: Array1<f32> = arr1(&[1f32, 1f32]);
        input = input.exp();
        let target: Array1<f32> = arr1(&[2.7182817, 2.7182817]);
        abs_diff_eq!(input, target, epsilon=1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let mut input: Array2<f64> = arr2(&[[1f64, 1f64], [1f64, 1f64]]);
        let target: Array2<f64> = arr2(&[
            [0.7310585786,0.7310585786],
            [0.7310585786,0.7310585786]]);
        input = input.sigmoid();
        abs_diff_eq!(input, target, epsilon=1e-6);
    }
}
