#![allow(dead_code, unused_imports)]

use std::collections::HashSet;

use hdf5;
use ndarray::{
    parallel::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    Array, ArrayView2, Axis,
};
use simct::{CoverTree, CoverTreeBuilder, Metric};

fn main() {
    let arg: Vec<_> = std::env::args().collect();

    let chunk_size = if arg.len() >= 2 {
        arg[1].to_string().parse::<usize>().unwrap()
    } else {
        1000
    };

    let file =
        // hdf5::File::open("./data/glove-25-angular.hdf5")
        hdf5::File::open("./data/sift-128-euclidean.hdf5").unwrap();
    let train = file.dataset("train").unwrap().read_2d::<f64>().unwrap();
    println!("{}", train.nrows());

    let ct = CoverTreeBuilder::new()
        .depth(15)
        .metric(Metric::Euclidean)
        .chunk_size(chunk_size)
        .build(train);

    assert!(ct.is_some());
    let ct = ct.unwrap();
    ct.verify();

    println!("{}", ct.count());
    println!("{}", ct.sum());

    let test = file.dataset("test").unwrap().read_2d::<f64>().unwrap();
    let validation = file
        .dataset("neighbors")
        .unwrap()
        .read_2d::<usize>()
        .unwrap();

    validate(&ct, 3281, test.view(), validation.view());
}

fn validate(ct: &CoverTree, idx: usize, test: ArrayView2<f64>, validation_mtx: ArrayView2<usize>) {
    let mut vs = HashSet::new();
    for x in validation_mtx.row(idx) {
        vs.insert(*x);
    }

    dbg!(&vs);

    let mut count = 0;
    let result = ct.search(test.row(idx), 100);
    for nn in &result {
        if vs.contains(&nn.idx()) && nn.idx() == 237424 {
            count += 1;
        }
    }

    println!("{} > count = {}/{}", idx, count, vs.len());
}
