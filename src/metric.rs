use crate::Scalar;
use ndarray::ArrayView1;

use ndarray_stats::DeviationExt;

/// Enum for distance functions in a metric space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    /// L-1 or Manhattan distance. See [\[Wikipedia\]](https://en.wikipedia.org/wiki/Taxicab_geometry).
    Manhattan,
    /// L-2 or Euclidean distance. See [\[Wikipedia\]](https://en.wikipedia.org/wiki/Euclidean_distance)
    Euclidean,
    /// L-inf or Chebyshev distance. See [\[Wikipedia\]](https://en.wikipedia.org/wiki/Chebyshev_distance)
    Chebyshev,
    /// Angular or cosine distance
    Angular,
}

impl Metric {
    /// Calculate the distance between two points.
    pub fn distance(&self, a: ArrayView1<'_, Scalar>, b: ArrayView1<'_, Scalar>) -> Scalar {
        match self {
            Metric::Manhattan => a.l1_dist(&b).unwrap(),
            Metric::Euclidean => a.l2_dist(&b).unwrap() as Scalar,
            Metric::Chebyshev => a.linf_dist(&b).unwrap(),
            Metric::Angular => {
                let mut dot = a.dot(&b);
                // floating point issue (e.g. 1.0000000000000002).
                if dot > 1. {
                    dot = 1.;
                }

                1. - dot.acos() / (std::f64::consts::PI as Scalar)
            }
        }
    }
}
