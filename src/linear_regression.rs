use ndarray::Array2;
use ndarray_linalg::InverseInto;

// sigmoid関数を実装している
// $$ \sigmoid(a) = \frac{1}{1 + \exp(-a)} $$
pub fn sigmoid(x: & Array2<f64>) -> Array2<f64> {
    let x:Array2<f64> = x * -1f64;
    let x= x.mapv(f64::exp);
    // let ones = Array2::<f64, _>::ones(x.raw_dim().f());
    1f64 / (x + 1f64)
}

// pub fn gauss(x: Array2) -> Array {

// }

pub struct LinearRegression {
    pub input_dim: u32,
    pub base_func: String
}


impl LinearRegression {
    pub fn fit(&self, input: Array2<f64>, label: Array2<f64>) -> Array2<f64> {
        // φ(x)のを計算
        // 計画行列を計算
        if self.base_func == "Sigmoid" {
            let input = sigmoid(&input); 
        }

        // \bf{w}_{ML} = (\bf{\Phi}^T\bf{\Phi})^{-1} \Phi \bf{t}を計算
        // PRML p139 上巻
        let phi_t = input.t();
        let phi_t_phi = phi_t.dot(&input);
        let phi_t_phi_inv = phi_t_phi.inv_into().unwrap();
        phi_t_phi_inv.dot(&phi_t).dot(&label)
        // inv.dot(&PhiT)

    }

    pub fn predict(input:Array2<f64>, weight: Array2<f64>) -> Array2<f64> {
        weight.t().dot(&input)
    }
}

