# simct-rs

[![Crates.io](https://img.shields.io/crates/v/simct)](https://crates.io/crates/simct) [![Documentation](https://docs.rs/simct/badge.svg)](https://docs.rs/simct)

*Under development*

Nearest neighbour search with cover tree.


## Some benchmarks
For benchmarking, I compared runtime performance of my Rust implementation with the following algorithms in the test suite [ANN benchmarks](https://github.com/erikbern/ann-benchmarks) written by Erik Bernhardsson 
- sklearn's ball tree
- sklearn's kd tree
- spotify's annoy (also developed by Erik)

Actually, I wanted to run experiments with other known libraries (flann and faiss) but couldn't due to errors in installation (perhaps because of my pyenv). On the other hand, I managed to make hnswlib to run. However, after 20 minutes it hadn't finished building index while making my CPU completely busy and forced my to restart the PC (Ctrl+C didn't even free the consumsed resources... :(). 

The following command is used to run the experiments with the suite:

```batch
python3 run.py --definitions algos.yaml --local --dataset gist-960-euclidean --local -k 10 --batch
```

For simct-rs, I use the following settings to build the tree:
```batch
./target/release/examples/ann --file ./data/gist-960-euclidean.hdf5 --depth 20 --batch -k 10
```

All experiments are conducted five times on my PC Fedora 34 with AMD® Ryzen 7 3800x 8-core processor and 32 GB RAM. The result is as follows:

| Search 10 nearest neighbours | simct-rs |sklearn's ball tree  | sklearn kd tree | annoy
--- | --- | --- | --- | --- 
| Build time (seconds) | 196 |224|224|186 
| Search time / query (seconds) |0.217|0.249|0.256| 0.00014

Without a doubt, ```annoy``` is the best here with its blazingly fast query time. On the other hand, my Rust implementation is only faster than sklearn's CPython tree by a small amount. This means that the indexing process needs rework and optimisation in order to reduce the query time.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
