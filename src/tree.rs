use std::{
    ops::Div,
    sync::{Arc, RwLock},
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{norm, Norm, NormalizeAxis};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use crate::{Metric, Scalar};

#[derive(Debug)]
pub struct Neighbour {
    idx: usize,
    dist: Scalar,
    node: Arc<RwLock<Node>>,
}

impl Neighbour {
    pub fn idx(&self) -> usize {
        self.idx
    }

    pub fn dist(&self) -> Scalar {
        self.dist
    }
}

#[derive(Clone, Debug)]
pub struct CoverTree {
    base: Scalar,
    metric: Metric,
    root: Option<Arc<RwLock<Node>>>,
}

impl Default for CoverTree {
    fn default() -> Self {
        Self {
            base: 2.,
            metric: Metric::Euclidean,
            root: None,
        }
    }
}

impl CoverTree {
    pub fn verify(&self) {
        match &self.root {
            Some(root) => root.read().unwrap().verify(self.metric),
            None => {}
        };
    }

    pub fn count(&self) -> usize {
        match &self.root {
            Some(root) => root.read().unwrap().count() + 1,
            None => 0,
        }
    }

    pub fn sum(&self) -> usize {
        match &self.root {
            Some(root) => root.read().unwrap().sum(),
            None => 0,
        }
    }

    pub fn insert(&mut self, data: Array1<Scalar>) {
        let count = self.count();
        let node = Arc::new(RwLock::new(Node::new(count, self.base, data)));
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
                }

                if d_px > covdist {
                    {
                        let mut p = root.clone();
                        let nr = node.read().unwrap();

                        while d_px > 2. * covdist {
                            let opt = p.write().unwrap().find_rem_leaf();
                            match opt {
                                Some(leaf) => {
                                    {
                                        let mut lw = leaf.write().unwrap();
                                        let p_reader = p.read().unwrap();
                                        lw.update_level(p_reader.level + 1);
                                        lw.children.push(p.clone());
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
                    Self::add_child(&node, root.clone(), self.metric);
                    self.root = Some(node);
                } else {
                    Self::add_child(&root, node, self.metric);
                }
            }
            None => {
                self.root = Some(node);
            }
        }
    }

    fn add_child(parent: &Arc<RwLock<Node>>, child_node: Arc<RwLock<Node>>, metric: Metric) {
        let d_pc = {
            let pr = parent.read().unwrap();
            let d_pc = {
                let cr = child_node.read().unwrap();
                metric.distance(pr.data.view(), cr.data.view())
            };

            for q in &pr.children {
                let (d_qx, covdist) = {
                    let q_reader = q.read().unwrap();
                    let cr = child_node.read().unwrap();
                    let d_qx = metric.distance(q_reader.data.view(), cr.data.view());
                    (d_qx, q_reader.covdist)
                };

                if d_qx <= covdist {
                    return Self::add_child(q, child_node, metric);
                }
            }

            d_pc
        };

        let mut pw = parent.write().unwrap();

        {
            let mut cw = child_node.write().unwrap();
            cw.update_level(pw.level - 1);
        }

        pw.children.push(child_node);

        if d_pc > pw.maxdist {
            pw.maxdist = d_pc;
        }
    }

    pub fn search(&self, query: ArrayView1<Scalar>, k: usize) -> Vec<Neighbour> {
        match &self.root {
            Some(root) => {
                let mut result = if self.metric == Metric::Angular {
                    let q = query.div(query.norm());
                    Self::nn2(q.view(), root, self.metric, k)
                } else {
                    Self::nn2(query, root, self.metric, k)
                };

                result.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
                result.truncate(k);
                result
            }
            None => Vec::new(),
        }
    }

    fn nn2(
        x: ArrayView1<Scalar>,
        p: &Arc<RwLock<Node>>,
        metric: Metric,
        k: usize,
    ) -> Vec<Neighbour> {
        let mut nn = {
            let pr = p.read().unwrap();
            if pr.children.is_empty() {
                return Vec::with_capacity(0);
            }

            let mut nn: Vec<_> = pr
                .children
                .par_iter()
                .enumerate()
                .map(|(_idx, node)| {
                    let nr = node.read().unwrap();
                    let dta = nr.data.view();
                    let dist = metric.distance(x, dta);

                    Neighbour {
                        idx: nr.idx,
                        dist,
                        node: node.clone(),
                    }
                })
                .collect();

            nn.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            let bound = pr.covdist
                + match nn.get(k) {
                    Some(nb) => nb.dist,
                    None => nn.last().unwrap().dist,
                };

            nn.retain(|x| x.dist <= bound);
            nn
        };

        let mut nnc: Vec<_> = nn
            .par_iter()
            .map(|nb| Self::nn2(x.view(), &nb.node, metric, k))
            .flatten()
            .collect();

        nn.append(&mut nnc);

        nn
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
                panic!("Noooooooooo")
            }
        };

        if one_lesser {
            one.raise(level);
        } else {
            two.raise(level);
        }

        // let mut leftover = Self::merge_internal(one, two);
        let two_root = two.root.take().unwrap();
        let mut leftover = Self::merge_nodes(one.root.as_ref().unwrap(), two_root, one.metric);
        for node in leftover.drain(0..) {
            Self::add_child(one.root.as_ref().unwrap(), node, one.metric);
        }
    }

    fn merge_nodes(
        p: &Arc<RwLock<Node>>,
        q: Arc<RwLock<Node>>,
        metric: Metric,
    ) -> Vec<Arc<RwLock<Node>>> {
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
                        metric.distance(pr.data.view(), rr.data.view()),
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
                            metric.distance(rr.data.view(), sr.data.view())
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
                            leftover.append(&mut Self::merge_nodes(s, r, metric));
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

        Self::add_child(&p, q, metric);
        let mut leftover_prime = Vec::with_capacity(leftover.len());

        for r in leftover.drain(0..) {
            let (d_rp, covdist) = {
                let rr = r.read().unwrap();
                let pr = p.read().unwrap();
                (metric.distance(rr.data.view(), pr.data.view()), pr.covdist)
            };

            if d_rp < covdist {
                Self::add_child(&p, r, metric);
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
                        lw.update_level(new_level);
                        lw.children.push(root.clone());
                    }

                    root = leaf;
                }
                None => {
                    let mut root_writer = root.write().unwrap();
                    let new_level = root_writer.level + 1;
                    root_writer.update_level(new_level);
                }
            };

            current_lvl = root.read().unwrap().level;
        }
    }

    #[inline(always)]
    fn distance(&self, a: ArrayView1<Scalar>, b: ArrayView1<Scalar>) -> Scalar {
        self.metric.distance(a, b)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CoverTreeBuilder {
    base: Option<Scalar>,
    metric: Option<Metric>,
    depth: Option<usize>,
    chunk_size: Option<usize>,
}

impl CoverTreeBuilder {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn base(mut self, base: Scalar) -> Self {
        self.base = Some(base);
        self
    }

    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = Some(metric);
        self
    }

    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self
    }

    pub fn build(self, mut data: Array2<Scalar>) -> Option<CoverTree> {
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

        Some(ct)
    }

    fn build_internal(
        self,
        chunk_index: usize,
        data: ArrayView2<Scalar>,
        metric: Metric,
        max_dist: Scalar,
        min_dist: Scalar,
    ) -> CoverTree {
        let base = match self.base {
            Some(base) => base,
            None => match self.depth {
                Some(depth) => (2_f64).powf((max_dist / min_dist).log2() / depth as f64),
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

        // let node = Node::with_level(node_idx, level, base, data.row(0).to_owned());
        // ct.root = Some(Arc::new(RwLock::new(node)));

        for (ii, dta) in data.outer_iter().enumerate() {
            let node = Node::with_level(node_idx + ii, level, base, dta.to_owned());
            ct.insert_node(Arc::new(RwLock::new(node)));
        }

        ct
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    idx: usize,
    level: i32,
    base: Scalar,
    covdist: Scalar,
    maxdist: Scalar,
    data: Array1<Scalar>,
    children: Vec<Arc<RwLock<Node>>>,
}

impl Node {
    pub(crate) fn new(idx: usize, base: Scalar, data: Array1<Scalar>) -> Self {
        Self {
            idx,
            level: 0,
            base,
            covdist: 1.,
            maxdist: Scalar::MIN,
            data,
            children: Vec::with_capacity(8),
        }
    }

    pub(crate) fn with_level(idx: usize, level: i32, base: Scalar, data: Array1<Scalar>) -> Self {
        Self {
            idx,
            level,
            base,
            covdist: base.powi(level),
            maxdist: Scalar::MIN,
            data,
            children: Vec::with_capacity(8),
        }
    }

    pub(crate) fn update_level(&mut self, level: i32) {
        self.level = level;
        self.covdist = self.base.powi(level);
    }

    /// Find and remove a leaf.
    pub(crate) fn find_rem_leaf(&mut self) -> Option<Arc<RwLock<Node>>> {
        let mut idx = None;
        for (child_idx, child) in self.children.iter().enumerate() {
            let mut cr = child.write().unwrap();
            if cr.children.is_empty() {
                idx = Some(child_idx);
                break;
            } else if let Some(q) = cr.find_rem_leaf() {
                return Some(q);
            }
        }

        // Ordering among children are not important. So we use swap_remove for quick remove.
        match idx {
            Some(idx) => Some(self.children.swap_remove(idx)),
            None => None,
        }
    }

    pub fn verify(&self, metric: Metric) {
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

    pub fn count(&self) -> usize {
        let mut count = self.children.len();

        for q in &self.children {
            let qr = q.read().unwrap();
            count += qr.count();
        }

        count
    }

    pub fn sum(&self) -> usize {
        let mut sum = self.idx;

        for q in &self.children {
            let qr = q.read().unwrap();
            sum += qr.sum();
        }

        sum
    }
}
