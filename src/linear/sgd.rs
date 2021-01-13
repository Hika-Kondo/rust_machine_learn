use ndarray::{Array2, Array, ArrayView, Axis, ScalarOperand};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_stats::QuantileExt;

use crate::estimator::{Estimator, Learner};
use crate::traits::RMLType;
use crate::linear::{BasicFunc, preprocess};


#[derive(Clone)]
pub struct IterLinearRegression<T: RMLType> {
    basicfunc: BasicFunc,
    epoch: u32,
    lr: T,
}


impl<T: RMLType> IterLinearRegression::<T> {
    pub fn new(str: String, epoch: u32, lr: T) -> IterLinearRegression::<T> {
        let func = match &*str {
            "Sigmoid" => BasicFunc::Sigmoid,
            _ => BasicFunc::None,
        };
        IterLinearRegression {
            basicfunc: func,
            epoch: epoch,
            lr: lr,
        }
    }
}


pub struct IterLinearRegressionResult<T: RMLType> {
    weight: Array2<T>,
    config: IterLinearRegression::<T>,
}

impl<T: RMLType + ScalarOperand> Learner<T> for IterLinearRegression::<T> {
    type LearnedModel = IterLinearRegressionResult::<T>;
    type Input = Array2<T>;
    type Target = Array2<T>;
    fn fit(&self, input: Self::Input, target: Self::Target) -> Self::LearnedModel {
        let input = preprocess(self.basicfunc, input);

        // let mut weight = Array2::<T>::random((1, input.shape()[0]), Normal::<T>::new(1., 1.).unwrap());
        // let mut weight = Array2::<T>::random((1,15), Uniform::new(-10., 10.));
        let mut weight = Array::<T, _>::zeros((1,input.shape()[1]));
        for epoch in 0..self.epoch as usize {
            for idx in 0..input.shape()[0] as usize {
                let batch = input.index_axis(Axis(0), idx).insert_axis(Axis(0));
                let now_target = target.index_axis(Axis(0), idx).insert_axis(Axis(0));
                let weight_clone = weight.clone();
                let res = now_target.into_owned() - weight_clone.dot(&batch.t());
                let res = res.dot(&batch);
                weight = weight + res.mapv(|a| a * self.lr);
            }
        }
        IterLinearRegressionResult {
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
    fn test_iter() {
        let mode = "Sigmoid".to_string();
        let model = IterLinearRegression::new(mode, 100 as u32, 1e-3);
        let weight = Array::random((1,15), Normal::new(1.,1.).unwrap());
        let input = Array::random((10, 15), Normal::new(1.,1.).unwrap());
        let target = input.dot(&weight.t());
        println!("{:8.4}", target);
        println!("{:8.4}", weight);
        let res = model.fit(input, target);
        abs_diff_eq!(res.weight, weight);
        }
}
