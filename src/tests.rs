#![allow(unused_imports)]
use ndarray::array;

use crate::Metric;

#[test]
fn test_metric() {
    let arr1 = array![1., 2., 3., 4.];
    let arr2 = array![2., 3., 4., 5.];

    assert_eq!(
        4.,
        Metric::Manhattan.distance(arr1.view(), arr2.view()),
        "Test Manhattan distance"
    );
    assert_eq!(
        2.,
        Metric::Euclidean.distance(arr1.view(), arr2.view()),
        "Test Euclidean distance"
    );
    assert_eq!(
        1.,
        Metric::Chebyshev.distance(arr1.view(), arr2.view()),
        "Test Chebyshev distance"
    );
    // assert_eq!(
    //     40.,
    //     Metric::Angular.distance(arr1.view(), arr2.view()),
    //     "Test angular distance"
    // );
}
