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
        println!("{:?}", weight.shape());
        for epoch in 0..self.epoch as usize {
            for idx in 0..input.len() as usize {
                let batch = input.index_axis(Axis(0), idx).insert_axis(Axis(0));
                let now_target = target.index_axis(Axis(0), idx).insert_axis(Axis(0));
                let weight_t = weight.t();
                let res = now_target.into_owned() - weight_t.dot(&batch);
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

    #[test]
    fn test_iter() {
        let mode = "Sigmoid".to_string();
        let model = IterLinearRegression::new(mode, 100 as u32, 1e-3);
        let input: Array2<f64> = arr2(&[[1f64, 1f64], [2f64, 2f64], 
            [1f64, 3f64], [3f64, 5f64], [10f64, 29f64]]);
        let target: Array2<f64> = arr2(&[[2f64], [4f64], [4f64], [8f64], [39f64]]);
        let res = model.fit(input, target);
    }
}
