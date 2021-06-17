//! A library for approximate nearest neighbour search based on simplified cover tree.
//!
//! # Definition
//! A cover tree is a hierarchy of levels where each level covers the level directly under it. The
//! top-most of a tree contains only a single point which serves as the tree's root and the bottom-most
//! level consists of all points in the dataset.
//!
//! Each level of a cover tree is associated with an integer number, ```i```. In the *implicit*
//! representation, a cover tree can have an infinite number of levels, that is the level number
//! ```i``` can go from ```-inf``` to ```inf```. Even though it's very useful in constructing and proving
//! algorithms and theorems, implementing the implicit form of a cover tree is almost impossible.
//! In this crate, the *explicit* representation is used to implement the cover tree data structure,
//! where the level number of the top-most layer is defined and will be increased only when needed.
//!
//! Regardless of representations, all levels in a cover tree must always maintain the following invariants
//! [\[1\]](#1):
//! - **Nesting invariant**: all points at a level ```i``` must also belong to the level ```i-1``` beneath it.
//! - **Covering invariant**: For every node ```q``` at a level ```i-1```, there exists a node ```p```
//! at the upper level ```i``` such that ```d(p, q) <= 2^i``` and exactly one such ```p``` is a parent
//! for ```q```.
//! - **Separating invariant**:  For all pairs of nodes ```(p_1, p_2)``` at a level ```i```, the distance
//! between two points must satisfy the condition: ```d(p_1, p_2) > 2^i```.
//!
//! These invariants form the foundation for constructing a cover tree. However, due to the nesting
//! invariant, the space required for the data structure may be greately increased during the construction
//! process. Instead of using the aforementioned invariants for the implemention of cover tree, we opt
//! for a simplified version with few tweaks in the invariants as follows [\[2\]](#2):
//! - **Levelling invariant**: For every node ```p``` at a level ```i```, we define the function
//! ```level(p) = i```. Then, for each child ```q``` of ```p```, the levels between two points must
//! satisfy the condition ```level(q) = level(p) - 1```.
//! - **Covering invariant**: For every node ```p``` and for every child node ```q``` of ```p```, the following
//! condition must always hold: ```d(p, q) <= 2^level(p)```, where ```d(p, q)``` is the distance
//! between two points in a metric space.
//! - **Separating invariant**:  For every node ```p``` and for any distinct children ```q_1``` and
//! ```q_2``` of ```p```, the distance function between two children must always satisfy the condition:
//! ```d(q_1, q_2) > 2^(level(p) - 1)```.
//!
//! By using this simplified and explicit representation, we only need ```n``` points to construct
//! a dataset.
//!
//! # The choice of ```base``` in exponentiation
//! In all representations mentioned above, the exponential of the level number ```i``` with base ```2```
//! is used as a bound on the distance between two points. However, such choice for the base makes
//! cover trees become unfit for a wide range of problems. For example, a level ```i = 3``` can cover
//! a distance range of ```(4, 8]```. However, at level ```i = 11```, the range can be as wide as
//! ```(1024, 2048]```. We can observe that as the level number increases, the distance range a level
//! covers grows exponentially. As a consequence, a higher level consists of (too) many points while
//! lower levels are starving for points.
//!
//! To deal with this issue, it was suggested in the original paper that the ```base = 1.37``` should
//! be used in runtime. For this choice of base, a level ```i = 3``` approximately covers a range
//! ```(1.87, 2.57]```, and ```i = 11``` ```(23.29, 31.90]```. We can clearly see that, adjusting the value of base
//! can help us dampen the exponential growth in the cover distance at each level and thus makes
//! cover tree become more efficient in many situations.
//!
//! # Implementation
//! In this library, the (simplified) cover tree data structure is built upon two essential crates:
//! [```ndarray```](https://crates.io/crates/ndarray) for numerical computation and [```rayon```](https://crates.io/crates/rayon)
//! for parallel processing. The insertion of a node to a tree and the procedure to merge two trees
//! during parallel construction are implemented according to algorithm ```2``` and ```4``` in [\[2\]](#2),
//! respectively.
//!
//! As mentioned above, the ```base``` in exponentiation can affect greatly the performance of a cover
//! tree. This parameter can be set through the function [```CoverTreeBuilder::base```] or can
//! be derived by setting the depth of a tree through the function [```CoverTreeBuilder::depth```].
//! In case both ```base``` and ```depth``` are not set, the default value ```base = 1.37``` is used.
//!
//! Users can controll the parallelism in constructing a cover tree through the function [```CoverTreeBuilder::chunk_size```].
//! An input dataset will be divided into smaller subsets with size ```chunk_size``` and a cover tree
//! will be constructed for each subset. Then, these smaller trees will be grouped in pair and merged
//! together until there is only one big tree left. The default value for ```chunk_size``` is ```0```,
//! which implies no parallelism in construction.
//!
//! # Metric
//! In the current version, the following metrics are supported by the library:
//! - ```L-1``` metric: [```Metric::Manhattan```]
//! - ```L-2``` metric: [```Metric::Euclidean```]
//! - ```L-inf``` metric: [```Metric::Chebyshev```]
//!
//!
//! # Usage
//! Through the function [```CoverTree::new```] we can construct a new cover tree for a given ```base```
//! and ```metric```. A default cover tree (created from [```CoverTree::default```]) has ```base = 1.37```
//! and Euclidean metric. After creating a tree, we can invoke the function [```CoverTree::insert```]
//! to add a new point to the tree sequentially, as shown in the example below:
//!
//! ```rust
//! use ndarray::{Array, Array1};
//! use simct::{CoverTree, CoverTreeBuilder, Metric};
//!
//! let mut rng = oorandom::Rand64::new(0);
//! let data = Array::from_shape_simple_fn((100, 50), || rng.rand_float());
//!
//! // An empty cover tree.
//! let mut ct = CoverTree::new(1.37, Metric::Euclidean);
//!
//! // Adds points to the tree sequentially.
//! for row in data.outer_iter() {
//!     ct.insert(row.to_owned());
//! }
//! ```
//!
//! Adding points sequentially is generally slow. We can reduce the construction time dividing the
//! input dataset into smaller subsets and constructing cover trees for these subsets. These trees
//! will be merged together at the end to form a final tree, as shown in the example below:
//!
//! ```rust
//! use ndarray::{Array, Array1, Array2};
//! use simct::{CoverTreeBuilder, Metric};
//!
//! let mut rng = oorandom::Rand64::new(0);
//! let data = Array::from_shape_simple_fn((1000, 50), || rng.rand_float());
//! let mut ct = CoverTreeBuilder::new().metric(Metric::Euclidean).chunk_size(50).build(data);
//!
//! // Search 10 nearest neighbours for query.
//! let query = Array1::from_shape_simple_fn(50, || rng.rand_float());
//! let _ = ct.search(query.view(), 10);
//!
//! // Insert a point to the tree.
//! ct.insert(query);
//!
//! // Search 10 nearest neighbours for 10 query points.
//! let queries = Array2::from_shape_simple_fn((10, 50), || rng.rand_float());
//! let _ = ct.search2(queries.view(), 10);
//!
//! ```
//!
//! # References
//! <a id="1">\[1\]</a> A. Beygelzimer et al., Cover Trees for Nearest Neighbor. ICML 2006. [doi:10.1145/1143844.1143857](https://dl.acm.org/doi/10.1145/1143844.1143857)
//! [\[pdf\]](https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/covertree.pdf)
//!
//! <a id="2">\[2\]</a> M. Izbicki and C. Shelton, Faster Cover Trees. Proceedings of the 32nd
//! International Conference on Machine Learning 2015. [\[pdf\]](http://proceedings.mlr.press/v37/izbicki15.pdf)
//!
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
pub use tree::Neighbour;
pub use tree::QueryResult;
