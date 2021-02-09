use ndarray::Array2;
use crate::traits::RMLType;

#[macro_use]
mod regression;
mod sgd;

pub use regression::BasicLinearRegression;
pub use sgd::IterLinearRegression;
pub use regression::MultiDimLinearRegression;

// for cal_weight, BasicFunc, preprocess
use crate::estimator::Estimator;


pub struct LinearResult<T: RMLType> {
    weight:Array2<T>,
}

impl<T: RMLType> Estimator for LinearResult<T> {
    type Input = Array2<T>;
    type Output = Array2<T>;

    fn predict(&self, input: Self::Input) -> Self::Output {
        input.dot(&self.weight)
    }
}
