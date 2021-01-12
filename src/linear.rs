use ndarray_linalg::InverseInto;
use ndarray::Array2;

#[macro_use]
mod regression;
mod sgd;

pub use regression::BasicLinearRegression;
pub use sgd::IterLinearRegression;

// for cal_weight, BasicFunc, preprocess
use crate::traits::RMLType;
use crate::func::Sigmoid;

pub fn cal_weight<T: RMLType>(input: Array2<T>, target: Array2<T>) -> Array2<T>{
    // \bf{w}_{ML} = (\bf{\Phi}^T\bf{\Phi})^{-1} \Phi \bf{t}を計算
    // PRML p139 上巻
    let phi_t = input.t();
    let phi_t_phi = phi_t.dot(&input);
    let phi_t_phi_inv = phi_t_phi.inv_into().unwrap();
    phi_t_phi_inv.dot(&phi_t).dot(&target)

}

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
