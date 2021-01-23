use ndarray::Array2;
use crate::traits::RMLType;

#[macro_use]
mod regression;
mod sgd;

pub use regression::BasicLinearRegression;
pub use sgd::IterLinearRegression;

// for cal_weight, BasicFunc, preprocess
use crate::func::Sigmoid;
use crate::estimator::Estimator;


#[derive(Clone, Debug, Copy)]
pub enum BasicFunc {
    Sigmoid,
    None,
}

pub fn preprocess<T: RMLType>(func: BasicFunc, input: Array2<T>) -> Array2<T> {
    match func {
        BasicFunc::Sigmoid => input.sigmoid(),
        BasicFunc::None => input,
    }
}

pub struct LinearResult<T: RMLType> {
    weight:Array2<T>,
    basicfunc: BasicFunc
}

impl<T: RMLType> Estimator for LinearResult<T> {
    type Input = Array2<T>;
    type Output = Array2<T>;

    fn predict(&self, input: Self::Input) -> Self::Output {
        let input = preprocess(self.basicfunc, input);
        input.dot(&self.weight)
    }
}
