use ndarray::Array2;
use ndarray_linalg::InverseInto;
use ndarray_linalg::lapack::Lapack;
use ndarray_linalg::types::Scalar;

use crate::func::Sigmoid;
use crate::estimator::{Estimator, Learner};

#[derive(Clone, Debug, Copy)]
enum BasicFunc {
    Sigmoid,
    Gauss,
    None,
}

fn preprocess<T: Scalar + Lapack>(func: BasicFunc, input: Array2<T>) -> Array2<T> {
    match func {
        BasicFunc::Sigmoid => input.sigmoid(),
        BasicFunc::Gauss => input,
        BasicFunc::None => input,
    }
}


#[derive(Clone)]
pub struct BasicLinearRegression {
    basicfunc: BasicFunc,
}

impl BasicLinearRegression {

    pub fn new(str: String) -> Self {
        let func = match &*str {
            "Sigmoid" => BasicFunc::Sigmoid,
            "Gauss" => BasicFunc::Gauss,
            _ => BasicFunc::None,
        };
        BasicLinearRegression {
            basicfunc: func
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
        let input = preprocess(self.basicfunc, input);
        // let input = match self.basicfunc {
        //     BasicFunc::Sigmoid => input.sigmoid(),
        //     BasicFunc::Gauss => input,
        //     _ => input,
        // };
        
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

// impl<T: Scalar + Lapack> Estimator<T> for BasicLinearRegressionResult<T> {
//     type EstimatorRes = BasicLinearRegressionResult::<T>;
//     type Input = Array2<T>;
    
//     fn predict(&self, input: Self::Input) -> Array2<T> {

//     }
// }


#[cfg(test)]
mod test {
    use super::*;
    use ndarray::*;
    use approx::abs_diff_eq;

    #[test]
    fn test_basic() {
        let mode = "Sigmoid".to_string();
        let model = BasicLinearRegression::new(mode);
        let input: Array2<f64> = arr2(&[[1f64, 1f64], [2f64, 2f64], 
            [1f64, 3f64], [3f64, 5f64], [10f64, 29f64]]);
        let target: Array2<f64> = arr2(&[[2f64], [4f64], [4f64], [8f64], [39f64]]);
        let res = model.fit(input, target);
        abs_diff_eq!(res.weight, arr2(&[[1f64], [1f64], [1f64]]));

        let mode = "Sigmoid".to_string();
        let model = BasicLinearRegression::new(mode);
        let input: Array2<f32> = arr2(&[[1f32, 1f32], [2f32, 2f32], 
            [1f32, 3f32], [3f32, 5f32], [10f32, 29f32]]);
        let target: Array2<f32> = arr2(&[[2f32], [4f32], [4f32], [8f32], [39f32]]);
        let res = model.fit(input, target);
        abs_diff_eq!(res.weight, arr2(&[[1f32], [1f32], [1f32]]));
    }
}
