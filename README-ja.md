# Rust machine learning crate
rustで実装した統計的な機械学習むけのクレートです。

# Features
`rust_ndarray`をもとに実装を行っております。
PRMLに掲載されている手法を当面は実装していこうと考えています。

# Futer Work
今後実装する予定のモデルです。

- [ ] 線型回帰
    - [x] linear regression
    - [x] iterative linear regression
    - [x] ロッソ回帰
    - [x] リッジかいき
    - [x] 複数次元
    - [ ] ベイズ線形回帰
- [ ] 線形識別
- [ ] ニューラルネットワーク
- [ ] カーネル法
- [ ] k-means
- [ ] 隠れマルコフモデル
- [ ] 主成分分析

# How To Use Cargo
```
[dependencies]
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
