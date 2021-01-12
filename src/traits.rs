use ndarray_linalg::lapack::Lapack;
use ndarray_linalg::types::Scalar;
// use num_traits::Float;
use ndarray_rand::rand_distr::Float as Float;

pub trait RMLType: Lapack + Scalar + Float {}

impl RMLType for f32 {}

impl RMLType for f64 {}
