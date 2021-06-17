#![allow(dead_code, unused_imports)]
use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use hdf5;
use ndarray::{
    parallel::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    s, Array, ArrayView2, Axis,
};
use ndarray_linalg::assert;
use simct::{CoverTree, CoverTreeBuilder, Metric, Neighbour};

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

    println!("Open file: {}", filepath);
    let (train, test) = {
        let file = hdf5::File::open(filepath).unwrap();
        let train = file.dataset("train").unwrap().read_2d::<f64>().unwrap();
        let test = file.dataset("test").unwrap().read_2d::<f64>().unwrap();
        let _validation = file
            .dataset("neighbors")
            .unwrap()
            .read_2d::<usize>()
            .unwrap();

        (train, test)
    };

    println!("> Dim: {} x {}", train.nrows(), train.ncols());
    println!("{}", std::mem::size_of::<Neighbour>());

    let start = Instant::now();
    let ct = CoverTreeBuilder::new()
        .depth(15)
        .metric(Metric::Euclidean)
        .chunk_size(chunk_size)
        .build(train);

    let build_duration = start.elapsed();
    println!(
        "> Construction duration: {:?} (ms)",
        build_duration.as_millis()
    );

    let start = Instant::now();
    let result = ct.search2(test.view(), 100);
    println!(" >>> {}", result.len());

    let search_duration = start.elapsed();
    println!("> Search duration: {:?} (ms)", search_duration.as_millis());
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
