#[test]
fn sgd() {
    use rust_machine_learning::linear::IterLinearRegression;
    let _model = IterLinearRegression::new("Sigmoid".to_string(), 100, 1e-3, 0.);
}
