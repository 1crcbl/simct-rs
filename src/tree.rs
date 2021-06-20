use std::sync::Weak;
use std::{
    collections::VecDeque,
    ops::Div,
    sync::{Arc, RwLock},
};

use std::iter::FromIterator;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{norm, Norm, NormalizeAxis};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{Metric, Scalar};

/// Query result containing k-nearest neighbours to a query point.
#[derive(Debug, Default)]
pub struct QueryResult {
    query_index: usize,
    neighbours: VecDeque<Neighbour>,
}

impl QueryResult {
    pub(crate) fn new(index: usize, neighbours: VecDeque<Neighbour>) -> Self {
        Self {
            query_index: index,
            neighbours,
        }
    }

    /// Returns the query index from batch query for this result.
    pub fn index(&self) -> usize {
        self.query_index
    }

    /// Sets the query index.
    pub fn set_index(&mut self, query_index: usize) {
        self.query_index = query_index;
    }

    /// Returns the nearest neighbours of a query.
    pub fn neighbours(&self) -> &VecDeque<Neighbour> {
        &self.neighbours
    }

    /// Consumes ```self``` and returns the query index and the nearest neighbours of that query.
    pub fn take(self) -> (usize, VecDeque<Neighbour>) {
        (self.query_index, self.neighbours)
    }
}

/// A neighbour resulted from a k-nearest neighbour search.
#[derive(Debug)]
pub struct Neighbour {
    idx: usize,
    query_idx: usize,
    dist: Scalar,
    node: Arc<RwLock<Node>>,
}

impl Neighbour {
    /// Returns the index of a neighbour.
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Returns the distance for a neighbour to a query point.
    pub fn dist(&self) -> Scalar {
        self.dist
    }
}

/// A type of tree data structure which is designed for fast nearest neighbour search in general
/// n-point metric spaces and requires ```O(n)``` space.
#[derive(Clone, Debug)]
pub struct CoverTree {
    base: Scalar,
    metric: Metric,
    root: Option<Arc<RwLock<Node>>>,
}

impl Default for CoverTree {
    /// Creates a default cover tree with ```base = 1.37``` and Euclidean metric.
    fn default() -> Self {
        Self {
            base: 1.37,
            metric: Metric::Euclidean,
            root: None,
        }
    }
}

impl CoverTree {
    /// Creates a new empty tree with given ```base``` and ```metric```.
    pub fn new(base: Scalar, metric: Metric) -> Self {
        Self {
            base,
            metric,
            root: None,
        }
    }

    /// Inserts a new point to a tree.
    pub fn insert(&mut self, data: Array1<Scalar>) {
        let count = self.size();
        let node = Arc::new(RwLock::new(Node::new(count, self.metric, self.base, data)));
        self.insert_node(node);
    }

    fn insert_node(&mut self, node: Arc<RwLock<Node>>) {
        match &self.root {
            Some(root) => {
                let (mut d_px, mut covdist, maxdist) = {
                    let reader = root.read().unwrap();
                    let nr = node.read().unwrap();

                    let d_px = self.distance(nr.data.view(), reader.data.view());
                    let maxdist = if d_px > reader.maxdist {
                        Some(d_px)
                    } else {
                        None
                    };

                    (d_px, reader.covdist, maxdist)
                };

                if let Some(dist) = maxdist {
                    root.write().unwrap().maxdist = dist;
                    node.write().unwrap().maxdist = dist;
                }

                if d_px > covdist {
                    {
                        let mut p = root.clone();
                        let nr = node.read().unwrap();

                        while d_px > self.base * covdist {
                            let opt = p.write().unwrap().find_rem_leaf();
                            match opt {
                                Some(leaf) => {
                                    // leaf becomes a new root.
                                    {
                                        let mut pw = p.write().unwrap();
                                        pw.set_parent(Some(Arc::downgrade(&leaf)));
                                    }
                                    {
                                        let mut lw = leaf.write().unwrap();
                                        let p_reader = p.read().unwrap();
                                        lw.set_level(p_reader.level + 1);
                                        lw.children.push(p.clone());
                                        lw.set_parent(None);
                                    }

                                    p = leaf;
                                }
                                None => {
                                    let mut p_writer = p.write().unwrap();
                                    p_writer.level += 1;
                                    p_writer.covdist *= self.base;
                                }
                            };

                            d_px = self.distance(p.read().unwrap().data.view(), nr.data.view());
                            covdist = p.read().unwrap().covdist;
                        }
                    }
                    Self::add_child(&node, root.clone());
                    self.root = Some(node);
                } else {
                    Self::add_child(&root, node);
                }
            }
            None => {
                self.root = Some(node);
            }
        }
    }

    fn add_child(parent: &Arc<RwLock<Node>>, child_node: Arc<RwLock<Node>>) {
        let (d_pc, p_level) = {
            let pr = parent.read().unwrap();
            let d_pc = {
                let cr = child_node.read().unwrap();
                cr.metric.distance(pr.data.view(), cr.data.view())
            };

            for q in &pr.children {
                let (d_qx, covdist) = {
                    let q_reader = q.read().unwrap();
                    let cr = child_node.read().unwrap();
                    let d_qx = cr.metric.distance(q_reader.data.view(), cr.data.view());
                    (d_qx, q_reader.covdist)
                };

                if d_qx <= covdist {
                    return Self::add_child(q, child_node);
                }
            }

            (d_pc, pr.level)
        };

        {
            let mut cw = child_node.write().unwrap();
            cw.set_level(p_level - 1);
            cw.set_parent(Some(Arc::downgrade(&parent)));
        }

        {
            let mut pw = parent.write().unwrap();
            // pw.update_parent_maxdist(&child_node);
            pw.children.push(child_node);

            if d_pc > pw.maxdist {
                pw.maxdist = d_pc;
            }
        }
    }

    /// Performs the nearest neighbour search for a single query and returns ```k``` neighbours who
    /// are closest to the ```query``` point.
    pub fn search(&self, query: ArrayView1<'_, Scalar>, k: usize) -> QueryResult {
        match &self.root {
            Some(root) => {
                let result = Self::exe_search(root, query, 0, k, self.metric);
                QueryResult::new(0, VecDeque::from_iter(result.into_iter()))
            }
            None => QueryResult::default(),
        }
    }

    /// Performs the nearest neighbour search for an array of queries and returns ```k``` neighbours who
    /// are closest to the points in the query array.
    pub fn search2(&self, query: ArrayView2<'_, Scalar>, k: usize) -> Vec<QueryResult> {
        match &self.root {
            Some(root) => {
                // Simple calculation for chunk size
                let nrows = query.nrows();
                let chunk_size = if nrows > 1000 {
                    1000
                } else if nrows > 100 {
                    100
                } else {
                    10
                };

                let mut tmpr: Vec<Neighbour> = query
                    .axis_chunks_iter(Axis(0), chunk_size)
                    .into_par_iter()
                    .enumerate()
                    .map(|(chunk_idx, chunk)| {
                        let tmp = chunk_idx * chunk_size;
                        chunk
                            .outer_iter()
                            .enumerate()
                            .map(|(idx, q1)| {
                                let row_idx = tmp + idx;
                                Self::exe_search(root, q1, row_idx, k, self.metric)
                            })
                            .flatten()
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .collect();

                let mut result: Vec<QueryResult> = (0..query.nrows())
                    .map(|ii| QueryResult {
                        query_index: ii,
                        neighbours: VecDeque::with_capacity(100),
                    })
                    .collect();

                for nb in tmpr.drain(..) {
                    result[nb.query_idx].neighbours.push_back(nb);
                }

                result
            }
            None => Vec::with_capacity(0),
        }
    }

    #[inline(always)]
    fn exe_search(
        root: &Arc<RwLock<Node>>,
        query: ArrayView1<'_, Scalar>,
        query_index: usize,
        k: usize,
        metric: Metric,
    ) -> Vec<Neighbour> {
        let mut result = Vec::with_capacity(k);

        if metric == Metric::Angular {
            let q = query.div(query.norm());
            Self::nn(q.view(), query_index, root, metric, k, &mut result)
        } else {
            Self::nn(query, query_index, root, metric, k, &mut result)
        };

        result
    }

    /// Executes the nearest neighbour search at a parent node.
    fn nn(
        x: ArrayView1<'_, Scalar>,
        query_idx: usize,
        p: &Arc<RwLock<Node>>,
        metric: Metric,
        k: usize,
        result: &mut Vec<Neighbour>,
    ) {
        let mut nn = {
            let pr = p.read().unwrap();
            if pr.children.is_empty() {
                return;
            }

            let mut nn: Vec<_> = pr
                .children
                .iter()
                .map(|node| {
                    let nr = node.read().unwrap();
                    let dist = metric.distance(x, nr.data.view());

                    Neighbour {
                        idx: nr.idx,
                        query_idx,
                        dist,
                        node: node.clone(),
                    }
                })
                .collect();

            nn.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            nn
        };

        for child in nn.drain(..) {
            let mut insert_idx = None;
            for (r_idx, r_n) in result.iter().enumerate() {
                if child.dist < r_n.dist {
                    insert_idx = Some(r_idx);
                    break;
                }
            }

            let node = child.node.clone();
            let d_cx = child.dist;

            match insert_idx {
                Some(idx) => {
                    if result.len() == k {
                        result.pop();
                    }

                    result.insert(idx, child);
                }
                None => {
                    if result.len() < k {
                        result.push(child);
                    }
                }
            }

            if result.len() < k {
                Self::nn(x.view(), query_idx, &node, metric, k, result);
            } else {
                let maxdist = node.read().unwrap().maxdist;
                let k_dist = result.last().unwrap().dist;

                if k_dist + maxdist > d_cx {
                    Self::nn(x.view(), query_idx, &node, metric, k, result);
                }
            }
        }
    }

    fn merge(one: &mut Self, two: &mut Self) {
        let (one_lesser, level) = match (&one.root, &two.root) {
            (Some(oner), Some(twor)) => {
                let one_rdr = oner.read().unwrap();
                let two_rdr = twor.read().unwrap();

                if one_rdr.level < two_rdr.level {
                    (true, two_rdr.level)
                } else {
                    (false, one_rdr.level)
                }
            }
            (Some(_), None) => return,
            (None, Some(_)) => return,
            (None, None) => {
                panic!("Empty roots")
            }
        };

        if one_lesser {
            one.raise(level);
        } else {
            two.raise(level);
        }

        let two_root = two.root.take().unwrap();
        let mut leftover = Self::merge_nodes(one.root.as_ref().unwrap(), two_root);
        for node in leftover.drain(0..) {
            Self::add_child(one.root.as_ref().unwrap(), node);
        }
    }

    fn merge_nodes(p: &Arc<RwLock<Node>>, q: Arc<RwLock<Node>>) -> Vec<Arc<RwLock<Node>>> {
        let mut uncovered = Vec::with_capacity(8);
        let mut leftover = Vec::with_capacity(8);
        let mut sepcov = Vec::with_capacity(8);

        // phase 1
        {
            let mut q_writer = q.write().unwrap();

            for r in q_writer.children.drain(0..) {
                let (d_pr, covdist, sepdist) = {
                    let pr = p.read().unwrap();
                    let rr = r.read().unwrap();
                    let sepdist = pr.covdist / pr.base;

                    (
                        pr.metric.distance(pr.data.view(), rr.data.view()),
                        pr.covdist,
                        sepdist,
                    )
                };

                if d_pr < covdist {
                    // Equivalent to foundmatch in the paper: Some(idx) <=> foundmatch = true
                    let mut match_idx = None;
                    let pr = p.read().unwrap();

                    for (idx, s) in pr.children.iter().enumerate() {
                        let d_sr = {
                            let rr = (&r).read().unwrap();
                            let sr = s.read().unwrap();
                            rr.metric.distance(rr.data.view(), sr.data.view())
                        };

                        if d_sr < sepdist {
                            // leftover.append(&mut Self::merge_nodes(&s, r, metric));
                            match_idx = Some(idx);
                            break;
                        }
                    }

                    match match_idx {
                        Some(idx) => {
                            let s = pr.children.get(idx).unwrap();
                            leftover.append(&mut Self::merge_nodes(s, r));
                        }
                        None => {
                            sepcov.push(r);
                        }
                    }
                } else {
                    uncovered.push(r);
                }
            }
        }

        // phase 2
        {
            let mut pw = p.write().unwrap();
            pw.children.append(&mut sepcov);
        }

        Self::add_child(&p, q);
        let mut leftover_prime = Vec::with_capacity(leftover.len());

        for r in leftover.drain(0..) {
            let (d_rp, covdist) = {
                let rr = r.read().unwrap();
                let pr = p.read().unwrap();
                (
                    pr.metric.distance(rr.data.view(), pr.data.view()),
                    pr.covdist,
                )
            };

            if d_rp < covdist {
                Self::add_child(&p, r);
            } else {
                leftover_prime.push(r);
            }
        }

        leftover_prime.append(&mut uncovered);
        leftover_prime
    }

    #[inline]
    fn raise(&mut self, level: i32) {
        let mut root = match &self.root {
            Some(node) => node.clone(),
            None => panic!("Uninitialised tree"),
        };

        let mut current_lvl = { root.read().unwrap().level };

        while current_lvl < level {
            let opt = root.write().unwrap().find_rem_leaf();
            match opt {
                Some(leaf) => {
                    {
                        let mut lw = leaf.write().unwrap();
                        let root_reader = root.read().unwrap();
                        let new_level = root_reader.level + 1;
                        lw.set_level(new_level);
                        lw.children.push(root.clone());
                    }

                    root = leaf;
                }
                None => {
                    let mut root_writer = root.write().unwrap();
                    let new_level = root_writer.level + 1;
                    root_writer.set_level(new_level);
                }
            };

            current_lvl = root.read().unwrap().level;
        }
    }

    #[inline(always)]
    fn distance(&self, a: ArrayView1<'_, Scalar>, b: ArrayView1<'_, Scalar>) -> Scalar {
        self.metric.distance(a, b)
    }

    #[allow(dead_code)]
    pub(crate) fn verify(&self) {
        match &self.root {
            Some(root) => root.read().unwrap().verify(self.metric),
            None => {}
        };
    }

    /// Returns the number of points in a tree.
    pub fn size(&self) -> usize {
        match &self.root {
            Some(root) => root.read().unwrap().count() + 1,
            None => 0,
        }
    }
}

/// A build struct for initialising a new cover tree.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CoverTreeBuilder {
    base: Option<Scalar>,
    metric: Option<Metric>,
    depth: Option<usize>,
    chunk_size: Option<usize>,
}

impl CoverTreeBuilder {
    /// Creates a builder with default parameters.
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    /// Sets the ```base``` in exponentiation when calculating the covering distance (or invariant)
    /// of a level.
    pub fn base(mut self, base: Scalar) -> Self {
        self.base = Some(base);
        self
    }

    /// Sets the distance function for a tree.
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Sets the desired depth of a tree.
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = Some(depth);
        self
    }

    /// Sets the chunk size for parallel executions.
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self
    }

    /// Constructs a cover tree based on the given data and parameters.
    pub fn build(self, mut data: Array2<Scalar>) -> CoverTree {
        // let norms = data.map_axis(Axis(0), |x| x.norm_l2());
        let metric = match self.metric {
            Some(metric) => {
                if metric == Metric::Angular {
                    data = norm::normalize(data, NormalizeAxis::Row).0;
                }
                metric
            }
            None => Metric::Euclidean,
        };

        // Find initial level for root.
        let mut max_dist = Scalar::MIN;
        let mut min_dist = Scalar::MAX;
        for ii in 1..data.nrows() {
            let d = metric.distance(data.row(0), data.row(ii));
            if d > max_dist {
                max_dist = d;
            }

            if d < min_dist {
                min_dist = d;
            }
        }

        let chunk_size = self.chunk_size.unwrap_or(0);

        let ct = if chunk_size > 0 {
            let mut trees: Vec<CoverTree> = data
                .axis_chunks_iter(Axis(0), chunk_size)
                .into_par_iter()
                .enumerate()
                .map(|(cidx, chunk)| self.build_internal(cidx, chunk, metric, max_dist, min_dist))
                .collect();

            let mut len = trees.len();

            while len > 1 {
                trees = trees
                    .par_chunks_mut(2)
                    .map(|x| {
                        let (a, b) = x.split_at_mut(1);

                        if !b.is_empty() {
                            CoverTree::merge(&mut a[0], &mut b[0]);
                        }

                        std::mem::take(&mut x[0])
                    })
                    .collect();
                len = trees.len();
            }

            trees.pop().unwrap()
        } else {
            self.build_internal(0, data.view(), metric, max_dist, min_dist)
        };

        ct
    }

    fn build_internal(
        self,
        chunk_index: usize,
        data: ArrayView2<'_, Scalar>,
        metric: Metric,
        max_dist: Scalar,
        min_dist: Scalar,
    ) -> CoverTree {
        let base = match self.base {
            Some(base) => base,
            None => match self.depth {
                Some(depth) => (2 as Scalar).powf((max_dist / min_dist).log2() / depth as Scalar),
                None => 1.37,
            },
        };

        let mut ct = CoverTree {
            root: None,
            base,
            metric,
        };

        let level = Scalar::log(max_dist, base).ceil() as i32;
        let node_idx = chunk_index * data.nrows();

        for (ii, dta) in data.outer_iter().enumerate() {
            let node = Node::with_level(node_idx + ii, metric, level, base, dta.to_owned());
            ct.insert_node(Arc::new(RwLock::new(node)));
        }

        ct
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    idx: usize,
    level: i32,
    metric: Metric,
    base: Scalar,
    covdist: Scalar,
    maxdist: Scalar,
    parent: Option<Weak<RwLock<Node>>>,
    data: Array1<Scalar>,
    children: Vec<Arc<RwLock<Node>>>,
}

impl Node {
    pub(crate) fn new(idx: usize, metric: Metric, base: Scalar, data: Array1<Scalar>) -> Self {
        Self {
            idx,
            level: 0,
            metric,
            base,
            covdist: 1.,
            maxdist: Scalar::MIN,
            parent: None,
            data,
            children: Vec::with_capacity(8),
        }
    }

    pub(crate) fn with_level(
        idx: usize,
        metric: Metric,
        level: i32,
        base: Scalar,
        data: Array1<Scalar>,
    ) -> Self {
        Self {
            idx,
            level,
            metric,
            base,
            covdist: base.powi(level),
            maxdist: Scalar::MIN,
            parent: None,
            data,
            children: Vec::with_capacity(8),
        }
    }

    #[inline(always)]
    pub(crate) fn set_parent(&mut self, parent: Option<Weak<RwLock<Node>>>) {
        self.parent = parent;
    }

    pub(crate) fn set_level(&mut self, level: i32) {
        self.level = level;
        self.covdist = self.base.powi(level);

        for c in &self.children {
            let mut cw = c.write().unwrap();
            cw.set_level(level - 1);
        }
    }

    /// Finds and removes a leaf.
    pub(crate) fn find_rem_leaf(&mut self) -> Option<Arc<RwLock<Node>>> {
        let mut idx = None;
        for (child_idx, child) in self.children.iter().enumerate() {
            let mut cw = child.write().unwrap();
            if cw.children.is_empty() {
                idx = Some(child_idx);
                break;
            } else if let Some(q) = cw.find_rem_leaf() {
                return Some(q);
            }
        }

        // Ordering among children are not important. So we use swap_remove for quick remove.
        match idx {
            Some(idx) => {
                let leaf = self.children.swap_remove(idx);
                {
                    let mut lw = leaf.write().unwrap();
                    for child in &self.children {
                        let cr = child.read().unwrap();
                        let d = self.metric.distance(lw.data.view(), cr.data.view());

                        if d > lw.maxdist {
                            lw.maxdist = d;
                        }
                    }
                }
                Some(leaf)
            }
            None => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn verify(&self, metric: Metric) {
        if self.children.is_empty() {
            return;
        }

        for q1 in &self.children {
            let q1r = q1.read().unwrap();
            assert_eq!(self.level - 1, q1r.level);
            assert!(metric.distance(self.data.view(), q1r.data.view()) < self.covdist);

            let sepdist = self.covdist / self.base;

            for q2 in &self.children {
                if Arc::ptr_eq(q1, q2) {
                    continue;
                }

                let d_q1q2 = metric.distance(q1r.data.view(), q2.read().unwrap().data.view());
                assert!(d_q1q2 > sepdist);
            }
        }
    }

    pub(crate) fn count(&self) -> usize {
        let mut count = self.children.len();

        for q in &self.children {
            let qr = q.read().unwrap();
            count += qr.count();
        }

        count
    }
}
