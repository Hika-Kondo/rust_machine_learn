use ndarray::{Array2, Array1, Ix1, Ix2, ArrayBase};
use ndarray_linalg::InverseInto;
use ndarray_linalg::lapack::Lapack;
use ndarray_linalg::types::Scalar;
use ndarray::*;
use approx::abs_diff_eq;


#[test]
fn regression() {
    use rust_machine_learning::linear::basic::*;
    let model = BasicLinearRegression::new("Sigmoid".to_string());
}
