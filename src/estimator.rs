pub trait Learner<T> {
    type LearnedModel;
    type Input;
    type Target;
    fn fit(&self, input: Self::Input, target: Self::Target) -> Self::LearnedModel;
}

pub trait Estimator {
    type EstimatorRes;
    type Input;
    fn predict(&self, input: Self::Input) -> Self::EstimatorRes;
}
