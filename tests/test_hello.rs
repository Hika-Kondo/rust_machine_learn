extern crate zenu;

#[test]
fn test_hello() {
    assert_eq!(zenu::hello::hello(), "Hello".to_string());
}
