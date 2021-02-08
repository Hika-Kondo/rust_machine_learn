use ndarray::Array2;
use ndarray_linalg::InverseInto;

use crate::estimator::{Learner};
use crate::traits::RMLType;
use crate::linear::{BasicFunc, preprocess, LinearResult, Estimator};


#[derive(Clone)]
pub struct BasicLinearRegression {
    basicfunc: BasicFunc,
}

impl BasicLinearRegression {

    pub fn new(str: String) -> Self {
        let func = match &*str {
            "Sigmoid" => BasicFunc::Sigmoid,
            _ => BasicFunc::None,
        };
        BasicLinearRegression {
            basicfunc: func,
        }
    }

}

impl<T: RMLType> Learner<T> for BasicLinearRegression {
    type LearnedModel = LinearResult::<T>;
    type Input = Array2<T>;
    type Target = Array2<T>;
    fn fit(&self, input: Self::Input, target: Self::Target) -> Self::LearnedModel{
        let input = preprocess(self.basicfunc, input);
        
        let weight = cal_weight(input, target);
        
        LinearResult::<T> {
            weight: weight,
            basicfunc: self.basicfunc.clone(),
        }
    }
}

pub fn cal_weight<T: RMLType>(input: Array2<T>, target: Array2<T>) -> Array2<T>{
    // \bf{w}_{ML} = (\bf{\Phi}^T\bf{\Phi})^{-1} \Phi \bf{t}を計算
    // PRML p139 上巻
    let phi_t = input.t();
    let phi_t_phi = phi_t.dot(&input);
    let phi_t_phi_inv = phi_t_phi.inv_into().unwrap();
    phi_t_phi_inv.dot(&phi_t).dot(&target)

}


#[cfg(test)]
mod test {
    use super::*;
    use ndarray::*;
    use approx::abs_diff_eq;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    #[test]
    fn test_basic() {
        let mode = "Sigmoid".to_string();
        let model = BasicLinearRegression::new(mode);
        let weight = Array::random((1,15), Normal::new(1.,1.).unwrap());
        let input = Array::random((10, 15), Normal::new(1.,1.).unwrap());
        let target = input.dot(&weight.t());
        let res = model.fit(input, target);
        println!("regression is {:?}", res.weight.shape());
        println!("regression is {:?}", res.weight.shape());
        println!("regression is {:?}", res.weight.shape());
        let test_input = Array::random((12, 15), Normal::new(1.,1.).unwrap());
        let test_target = test_input.dot(&weight.t());
        let pred = res.predict(test_input);
        abs_diff_eq!(pred, test_target);
    }
}
