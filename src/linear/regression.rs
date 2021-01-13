use ndarray::Array2;

use crate::estimator::{Estimator, Learner};
use crate::traits::RMLType;
use crate::linear::{cal_weight, BasicFunc, preprocess};


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
    type LearnedModel = BasicLinearRegressionResult::<T>;
    type Input = Array2<T>;
    type Target = Array2<T>;
    fn fit(&self, input: Self::Input, target: Self::Target) -> Self::LearnedModel{
        let input = preprocess(self.basicfunc, input);
        
        let weight = cal_weight(input, target);
        
        BasicLinearRegressionResult::<T> {
            weight: weight,
            config: self.clone(),
        }
    }
}


pub struct BasicLinearRegressionResult<T: RMLType> {
    weight: Array2<T>,
    config: BasicLinearRegression,
}

impl<T: RMLType> Estimator for BasicLinearRegressionResult<T> {
    type Input = Array2<T>;
    type Output = Array2<T>;
    
    fn predict(&self, input: Self::Input) -> Self::Output {
        let input = preprocess(self.config.basicfunc, input);
        println!("{:?}", input.shape());
        input.dot(&self.weight)
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use ndarray::*;
    use approx::abs_diff_eq;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_stats::QuantileExt;

    #[test]
    fn test_basic() {
        let mode = "Sigmoid".to_string();
        let model = BasicLinearRegression::new(mode);
        let weight = Array::random((1,15), Normal::new(1.,1.).unwrap());
        let input = Array::random((10, 15), Normal::new(1.,1.).unwrap());
        let target = input.dot(&weight.t());
        let res = model.fit(input, target);
        let test_input = Array::random((12, 15), Normal::new(1.,1.).unwrap());
        let test_target = test_input.dot(&weight.t());
        let pred = res.predict(test_input);
        abs_diff_eq!(pred, test_target);
    }
}
