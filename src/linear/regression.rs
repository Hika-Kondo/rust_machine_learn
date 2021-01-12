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
        self.weight.dot(&input)
    }
}


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
        let test: Array2<f64> = arr2(&[[1f64, 2f64, 3f64]]);
        let pred_res: Array2<f64> = res.predict(test);
        abs_diff_eq!(pred_res, arr2(&[[6f64]]));

        let mode = "Sigmoid".to_string();
        let model = BasicLinearRegression::new(mode);
        let input: Array2<f32> = arr2(&[[1f32, 1f32], [2f32, 2f32], 
            [1f32, 3f32], [3f32, 5f32], [10f32, 29f32]]);
        let target: Array2<f32> = arr2(&[[2f32], [4f32], [4f32], [8f32], [39f32]]);
        let res = model.fit(input, target);
        abs_diff_eq!(res.weight, arr2(&[[1f32], [1f32], [1f32]]));
    }
}
