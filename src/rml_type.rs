use ndarray_linalg::lapack::Lapack;
use ndarray_linalg::types::Scalar;
use num_traits::Float;

pub trait RMLType: Lapack + Scalar + Float {}

impl RMLType for f32 {}

impl RMLType for f64 {}
