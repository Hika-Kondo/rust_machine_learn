use ndarray::{Array2, Array1, Ix1, Ix2, ArrayBase};
use ndarray_linalg::InverseInto;
use ndarray_linalg::lapack::Lapack;
use ndarray_linalg::types::Scalar;

use crate::func::Sigmoid;
use crate::estimator::{Estimator, Learner};

#[derive(Clone)]
enum BasicFunc {
    Sigmoid,
    Gauss,
    None,
}

#[derive(Clone)]
pub struct BasicLinearRegression {
    basicfunc: BasicFunc,
}

impl BasicLinearRegression {
    fn new(basic_func: String) -> Self {
        BasicLinearRegression {
            basicfunc: BasicFunc::Sigmoid,
        }
    }
}


pub struct BasicLinearRegressionResult<T: Scalar + Lapack> {
    weight: Array2<T>,
    config: BasicLinearRegression,
}


impl<T: Scalar + Lapack> Learner<T> for BasicLinearRegression {
    type LearnedModel = BasicLinearRegressionResult::<T>;
    type Input = Array2<T>;
    type Target = Array2<T>;
    fn fit(&self, input: Self::Input, target: Self::Target) -> Self::LearnedModel{
        let input = match self.basicfunc {
            BasicFunc::Sigmoid => input.sigmoid(),
            BasicFunc::Gauss => input,
            _ => input,
        };
        
        // \bf{w}_{ML} = (\bf{\Phi}^T\bf{\Phi})^{-1} \Phi \bf{t}を計算
        // PRML p139 上巻
        let phi_t = input.t();
        let phi_t_phi = phi_t.dot(&input);
        let phi_t_phi_inv = phi_t_phi.inv_into().unwrap();
        let weight = phi_t_phi_inv.dot(&phi_t).dot(&target);
        BasicLinearRegressionResult::<T> {
            weight: weight,
            config: self.clone(),
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use ndarray::*;
    use approx::abs_diff_eq;

    #[test]
    fn test_basic() {
        let model = BasicLinearRegression::new("Sigmoid".to_string());
        let input: Array2<f64> = arr2(&[[1f64, 1f64], [2f64, 2f64], [1f64, 3f64], [3f64, 5f64], [10f64, 29f64]]);
        let target: Array2<f64> = arr2(&[[2f64], [4f64], [4f64], [8f64], [39f64]]);
        let res = model.fit(input, target);
        abs_diff_eq!(res.weight, arr2(&[[1f64], [1f64], [1f64]]));

        let model = BasicLinearRegression::new("Sigmoid".to_string());
        let input: Array2<f32> = arr2(&[[1f32, 1], [2, 2f], [1, 3], [3, 5], [10, 29]]);
        let target: Array2<f32> = arr2(&[[2f32], [4], [4], [8], [39]]);
        let res = model.fit(input, target);
        abs_diff_eq!(res.weight, arr2(&[[1f32], [1f32], [1f32]]));
    }
}
