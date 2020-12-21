extern crate zenu;

use ndarray::*;
use ndarray::Array2;

#[test]
fn test_linear_regression() {
    let regression_conf = zenu::linear_regression::LinearRegression{
        input_dim: 32, base_func: "Sigmoid".to_string()
    };
    let input: Array2<f64> = arr2(&[[1f64, 1f64], [2f64, 2f64], [1f64, 3f64], [3f64, 5f64], [10f64, 29f64]]);
    let target: Array2<f64> = arr2(&[[2f64], [4f64], [4f64], [8f64], [39f64]]);
    let weight = regression_conf.fit(input, target);
    assert_eq!(arr2(&[[1f64], [1f64]]).raw_dim(), weight.raw_dim());
}
