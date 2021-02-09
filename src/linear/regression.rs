use ndarray::Array2;
use ndarray_linalg::InverseInto;

use crate::estimator::{Learner};
use crate::traits::RMLType;
use crate::linear::LinearResult;


pub fn cal_weight<T: RMLType>(input: Array2<T>, target: Array2<T>) -> Array2<T>{
    // \bf{w}_{ML} = (\bf{\Phi}^T\bf{\Phi})^{-1} \Phi \bf{t}を計算
    // PRML p139 上巻
    let phi_t = input.t();
    let phi_t_phi = phi_t.dot(&input);
    let phi_t_phi_inv = phi_t_phi.inv_into().unwrap();
    phi_t_phi_inv.dot(&phi_t).dot(&target)

}


#[derive(Clone)]
pub struct BasicLinearRegression;

impl BasicLinearRegression {

    pub fn new() -> Self {
        BasicLinearRegression
    }

}

impl<T: RMLType> Learner<T> for BasicLinearRegression {
    type LearnedModel = LinearResult::<T>;
    type Input = Array2<T>;
    type Target = Array2<T>;
    fn fit(&self, input: Self::Input, target: Self::Target) -> Self::LearnedModel{
        
        let weight = cal_weight(input, target);
        
        LinearResult::<T> {
            weight: weight,
        }
    }
}


#[derive(Clone)]
pub struct MultiDimLinearRegression;

impl MultiDimLinearRegression {
    pub fn new() -> Self {
        MultiDimLinearRegression
    }
}


impl<T: RMLType> Learner<T> for MultiDimLinearRegression {
    type LearnedModel = LinearResult::<T>;
    type Input = Array2<T>;
    type Target = Array2<T>;
    fn fit(&self, input: Self::Input, target: Self::Target) -> Self::LearnedModel{
        let weight = cal_weight(input, target);
        
        LinearResult::<T> {
            weight: weight,
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use ndarray::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    use approx::assert_abs_diff_eq;

    use crate::linear::Estimator;

    #[test]
    fn test_basic() {
        let model = BasicLinearRegression::new();
        let weight = Array::random((1,15), Normal::new(1.,1.).unwrap());
        let input = Array::random((1000, 15), Normal::new(1.,1.).unwrap());
        let target = input.dot(&weight.t());
        let res = model.fit(input, target);
        println!("regression is {:?}", res.weight.shape());
        println!("regression is {:?}", res.weight.shape());
        println!("regression is {:?}", res.weight.shape());
        let test_input = Array::random((12, 15), Normal::new(1.,1.).unwrap());
        let test_target = test_input.dot(&weight.t());
        let pred = res.predict(test_input);
        assert_abs_diff_eq!(pred, test_target, epsilon=1e-3);
    }

    #[test]
    fn test_multi() {
        let model = MultiDimLinearRegression::new();
        let weight = Array::random((12,15), Normal::new(1.,1.).unwrap());
        let input = Array2::random((1000, 15), Normal::new(1.,1.).unwrap());
        let target = input.dot(&weight.t());
        let res = model.fit(input, target);
        let test_input = Array::random((12, 15), Normal::new(1.,1.).unwrap());
        let test_target = test_input.dot(&weight.t());
        let pred = res.predict(test_input);
        assert_abs_diff_eq!(pred, test_target, epsilon=1e-3);
    }
}
