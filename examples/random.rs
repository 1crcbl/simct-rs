use ndarray::{Array, Array1, Array2};
use simct::{CoverTreeBuilder, Metric};

// In this example, we generate a random array of 10000 points in a 50-dimensional Euclidean space.
// This array is used to construct a cover tree.
fn main() {
    let mut rng = oorandom::Rand64::new(0);
    let data = Array::from_shape_simple_fn((1000, 50), || rng.rand_float());
    let mut ct = CoverTreeBuilder::new()
        .metric(Metric::Euclidean)
        .chunk_size(50)
        .build(data);

    // Search 10 nearest neighbours for query.
    let query = Array1::from_shape_simple_fn(50, || rng.rand_float());
    let _ = ct.search(query.view(), 10);

    // Insert a point to the tree.
    ct.insert(query);

    // Search 10 nearest neighbours for 100 query points.
    let queries = Array2::from_shape_simple_fn((10, 50), || rng.rand_float());
    let _ = ct.search2(queries.view(), 10);
}
