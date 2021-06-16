use crate::Scalar;
use ndarray::ArrayView1;

use ndarray_stats::DeviationExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    /// L-1
    Manhattan,
    /// L-2
    Euclidean,
    /// L-inf
    Chebyshev,
    /// Angular/Cosine
    Angular,
}

impl Metric {
    pub fn distance(&self, a: ArrayView1<Scalar>, b: ArrayView1<Scalar>) -> Scalar {
        match self {
            Metric::Manhattan => a.l1_dist(&b).unwrap(),
            Metric::Euclidean => a.l2_dist(&b).unwrap(),
            Metric::Chebyshev => a.linf_dist(&b).unwrap(),
            Metric::Angular => {
                let mut dot = a.dot(&b);
                // floating point issue (e.g. 1.0000000000000002).
                if dot > 1. {
                    dot = 1.;
                }

                1. - dot.acos() / std::f64::consts::PI
            }
        }
    }
}
