# Rust machine learning crate
A crate for statistical machine learning implemented in rust.
This file is a translation of README-ja.md by deepl.

# Features
The implementation is based on `rust_ndarray`.
It is based on `rust_ndarray`.

# Futer Work
This is a model that we plan to implement in the future.

- [ ] linear regression
    - [x] linear regression
    - [x] iterative linear regression
    - [ ] Rosso regression 
- [ ] linear discrimination
- [ ] neural network
- [ ] Kernel method
- [ ] k-means
- [ ] Hidden Markov models
- [ ] Principal Component Analysis

# How To Use Cargo
````
[dependencies].
rust_machine_learning = { git = "https://github.com/bokutotu/rust_machine_learn" }
```

## dependencies
* ndarray = "0.13.0"
* ndarray-rand = "0.11.0"
* ndarray-linalg = { version = "0.12.1", features = ["intel-mkl"] }
* ndarray-stats = "0.3"
* num-traits = "0.2.14"
* approx = "0.3.1"
* rand = "0.8.1"

Translated with www.DeepL.com/Translator (free version)