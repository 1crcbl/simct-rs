#![allow(dead_code, unused_imports)]

use std::collections::HashSet;

use hdf5;
use ndarray::{
    parallel::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    s, Array, ArrayView2, Axis,
};
use simct::{CoverTree, CoverTreeBuilder, Metric};

fn main() {
    let arg: Vec<_> = std::env::args().collect();

    let (filepath, chunk_size) = if arg.len() >= 3 {
        (
            arg[1].to_string(),
            arg[2].to_string().parse::<usize>().unwrap(),
        )
    } else {
        eprintln!("The example requires two arguments: <file path> <chunk size>");
        std::process::exit(1);
    };

    let file = hdf5::File::open(filepath).unwrap();
    let train = file.dataset("train").unwrap().read_2d::<f64>().unwrap();
    println!("{}", train.nrows());

    let ct = CoverTreeBuilder::new()
        .depth(15)
        .metric(Metric::Euclidean)
        .chunk_size(chunk_size)
        .build(train);

    let test = file.dataset("test").unwrap().read_2d::<f64>().unwrap();
    let _validation = file
        .dataset("neighbors")
        .unwrap()
        .read_2d::<usize>()
        .unwrap();

    let _ = ct.search2(test.slice(s!(0..10, ..)), 100);
}

fn validate(ct: &CoverTree, idx: usize, test: ArrayView2<f64>, validation_mtx: ArrayView2<usize>) {
    let mut vs = HashSet::new();
    for x in validation_mtx.row(idx) {
        vs.insert(*x);
    }

    let mut count = 0;
    let result = ct.search(test.row(idx), 100);
    for nn in &result {
        if vs.contains(&nn.idx()) {
            count += 1;
        }
    }

    println!("{} > matched neighbours = {}/{}", idx, count, vs.len());
}
