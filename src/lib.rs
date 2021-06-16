//! A library for approximate nearest neighbour search based on simplified cover tree.
#![warn(
    missing_docs,
    rust_2018_idioms,
    missing_debug_implementations,
    broken_intra_doc_links
)]

type Scalar = f64;

mod metric;
pub use metric::Metric;

mod tests;

mod tree;
pub use tree::CoverTree;
pub use tree::CoverTreeBuilder;
