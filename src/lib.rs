type Scalar = f64;

mod metric;
pub use metric::Metric;

pub mod error;

mod tests;

mod tree;
pub use tree::CoverTree;
pub use tree::CoverTreeBuilder;
