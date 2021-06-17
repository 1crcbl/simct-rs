use ndarray::{Array, Array1, Array2};
use simct::{CoverTree, CoverTreeBuilder, Metric};

// In this example, we generate a random array of 10000 points in a 50-dimensional Euclidean space.
// We then pass the array to the builder to construct a cover tree. With chunk size is set to 50,
// the builder will build a tree for each 50 rows of the input array in parallel. Theese smaller
// trees will be merged with each other to create a final big tree consisting of all input points.
fn parallel() {
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

    // Search 10 nearest neighbours for 10 query points.
    let queries = Array2::from_shape_simple_fn((10, 50), || rng.rand_float());
    let _ = ct.search2(queries.view(), 10);
}

// We can also initialise an empty cover tree and add points to the tree sequentially.
fn sequential() {
    let mut rng = oorandom::Rand64::new(0);
    let data = Array::from_shape_simple_fn((100, 50), || rng.rand_float());

    // An empty cover tree.
    let mut ct = CoverTree::new(1.37, Metric::Euclidean);

    // Adds points to the tree sequentially.
    for row in data.outer_iter() {
        ct.insert(row.to_owned());
    }
}

fn main() {
    parallel();
    sequential();
}
