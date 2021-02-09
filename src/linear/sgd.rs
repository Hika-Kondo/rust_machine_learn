use ndarray::{Array2, Array , Axis, ScalarOperand};
use ndarray::parallel;
use ndarray_rand::rand_distr::Normal;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_linalg::norm::Norm;
use ndarray_linalg::types::Scalar;

use crate::estimator::{Learner};
use crate::traits::RMLType;
use crate::linear::{BasicFunc, preprocess, LinearResult, Estimator};


#[derive(Clone)]
pub struct IterLinearRegression<T: RMLType> {
    basicfunc: BasicFunc,
    epoch: u32,
    lr: T,
    lasso: T,
    ridge: T,
}


impl<T: RMLType> IterLinearRegression::<T> {
    pub fn new(str: String, epoch: u32, lr: T, lasso: T, ridge: T) -> IterLinearRegression::<T> {
        let func = match &*str {
            "Sigmoid" => BasicFunc::Sigmoid,
            _ => BasicFunc::None,
        };
        IterLinearRegression {
            basicfunc: func,
            epoch: epoch,
            lr: lr,
            lasso: lasso,
            ridge: ridge
        }
    }
}


impl<T: RMLType + ScalarOperand + Scalar<Real = T>> Learner<T> for IterLinearRegression::<T> {
    type LearnedModel = LinearResult::<T>;
    type Input = Array2<T>;
    type Target = Array2<T>;
    fn fit(&self, input: Self::Input, target: Self::Target) -> Self::LearnedModel {
        let input = preprocess(self.basicfunc, input);

        // let mut weight = Array2::<T>::random((1, input.shape()[0]), Normal::<T>::new(1., 1.).unwrap());
        // let mut weight = Array2::<T>::random((1,15), Uniform::new(-10., 10.));
        let mut weight = Array::<T, _>::zeros((1,input.shape()[1]));
        for _ in 0..self.epoch as usize {
            for idx in 0..input.shape()[0] as usize {
                let batch = input.index_axis(Axis(0), idx).insert_axis(Axis(0));
                let now_target = target.index_axis(Axis(0), idx).insert_axis(Axis(0));
                let weight_clone = weight.clone();
                let res = now_target.into_owned() - weight_clone.dot(&batch.t());
                let res = res.dot(&batch);
                weight = weight + res.mapv(|a| a * self.lr) + self.lasso * res.norm_l1() + self.ridge * res.norm_l2()

            }
        }
        Self::LearnedModel {
            weight: weight.t().into_owned(),
            basicfunc: self.basicfunc.clone(),
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_iter() {
        let mode = "_Sigmoid".to_string();
        let model = IterLinearRegression::new(mode, 1000 as u32, 1e-3, 1e-8, 0.);
        let weight = Array::random((1,15), Normal::new(0.,1.).unwrap());
        let input = Array::random((100, 15), Normal::new(0.,1.).unwrap());
        let target = input.dot(&weight.t());
        let res = model.fit(input, target);
        println!("sgd weight is {:?}", res.weight.shape());
        let test_input = Array::random((12, 15), Normal::new(0.,1.).unwrap());
        let test_target = test_input.dot(&weight.t());
        let pred = res.predict(test_input);
        println!("{:?}", pred);
        println!("{:?}", test_target);
        assert_abs_diff_eq!(pred, test_target, epsilon=1e-3);
        }
}
