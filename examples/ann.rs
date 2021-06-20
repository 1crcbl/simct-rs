#![allow(dead_code)]
use std::collections::HashSet;
use std::time::Instant;

use clap::{App, Arg};
use hdf5;
use ndarray::s;
use simct::{CoverTreeBuilder, Metric};

fn main() {
    let matches = App::new("simct benchmark")
        .arg(
            Arg::with_name("file")
                .short("f")
                .long("file")
                .takes_value(true)
                .required(true)
                .help("Path to a hdf5 dataset"),
        )
        .arg(
            Arg::with_name("chunk-size")
                .short("c")
                .long("chunk-size")
                .takes_value(true)
                .default_value("0")
                .help("Chunk size for parallel tree construction. Chunk size 0 means no parallel."),
        )
        .arg(
            Arg::with_name("base")
            .short("b")
            .long("base")
            .takes_value(true)
            .help("Base number in exponentiation")
        )
        .arg(
            Arg::with_name("depth")
            .short("d")
            .long("depth")
            .takes_value(true)
            .default_value("15")
            .help("Expected depth of a tree during construction. If base is also given, this parameter will be ignored.")
        )
        .arg(
            Arg::with_name("batch")
            .long("batch")
            .help("Flag to indicate whether a tree receives all queries at once or not. Default is false.")
        )
        .arg(
            Arg::with_name("runs")
                .long("runs")
                .takes_value(true)
                .default_value("5")
                .help("Number of runs for search query."),
        
        )
        .arg(
            Arg::with_name("k")
                .short("k")
                .takes_value(true)
                .default_value("10")
                .help("Number of nearest neighbours to search.")
        )
        .get_matches();

    let filepath = match matches.value_of("file") {
        Some(fp) => fp,
        None => {
            std::process::exit(1);
        }
    };

    let chunk_size = matches
        .value_of("chunk-size")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    println!("Chunk size: {}", chunk_size);

    let mut builder = CoverTreeBuilder::new().chunk_size(chunk_size);
    builder = match matches.value_of("base") {
        Some(base) => {
            let b = base.to_string().parse::<f64>().unwrap();
            println!("Base: {}", b);
            builder.base(b)
        }
        None => {
            let depth = match matches.value_of("depth") {
                Some(depth) => depth.to_string().parse::<usize>().unwrap(),
                None => 15,
            };

            println!("Depth: {}", depth);
            builder.depth(depth)
        }
    };

    println!("Open file: {}", filepath);
    let (train, test, neighbours) = {
        let file = hdf5::File::open(filepath).unwrap();
        let train = file.dataset("train").unwrap().read_2d::<f64>().unwrap();
        let test = file.dataset("test").unwrap().read_2d::<f64>().unwrap();
        let neighbours = file
            .dataset("neighbors")
            .unwrap()
            .read_2d::<usize>()
            .unwrap();

        (train, test, neighbours)
    };

    println!(
        "> Train      dimension: {} x {}",
        train.nrows(),
        train.ncols()
    );
    println!(
        "> Test       dimension: {} x {}",
        test.nrows(),
        test.ncols()
    );
    println!(
        "> Neighbour  dimension: {} x {}",
        neighbours.nrows(),
        neighbours.ncols()
    );
    println!("");

    println!("Constructing a cover tree from the train dataset...");
    let start = Instant::now();
    let ct = builder.metric(Metric::Euclidean).build(train);

    let build_duration = start.elapsed();
    println!(
        "> Construction duration: {:?} (ms)",
        build_duration.as_millis()
    );

    let runs = matches.value_of("runs").unwrap().to_string().parse::<usize>().unwrap();
    let k = matches.value_of("k").unwrap().to_string().parse::<usize>().unwrap();
    let n = neighbours.nrows();
    let total = n * k;

    for ii in 0..runs {
        println!("Run {}/{}", ii + 1, runs);
        let start = Instant::now();
    
        let results = if matches.is_present("batch") {
            println!(">Parallel query...");
            ct.search2(test.view(), k)
            // ct.search2(test.slice(s![0..1, ..]), k)
        } else {
            println!("> Sequential query...");
            let mut results = Vec::with_capacity(n);
            for (idx, row) in test.outer_iter().enumerate() {
                let mut qr = ct.search(row, k);
                qr.set_index(idx);
                results.push(qr);
            }
            results
        };
    
        println!("> Validating results...");
        let mut val_results = vec![0; n];
        let mut count = 0;
        for qr in results {
            let idx = qr.index();
            let row = neighbours.slice(s![idx, ..]);
            let mut hs = HashSet::with_capacity(k);
            for el in row.iter() {
                hs.insert(el);
            }
    
            for nb in qr.neighbours() {
                if hs.contains(&nb.index()) {
                    count += 1;
                }
    
                hs.remove(&nb.index());
            }
    
            val_results[idx] = count;
        }

        println!("> Score: {}/{}", count, total);
    
        let search_duration = start.elapsed();
        println!("> Total search duration: {:?} (ms)", search_duration.as_millis());
    }
}
