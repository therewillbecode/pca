mod lib;

use pca::*;

fn main() {
    let mut random_matrix = lib::rand_matrix();
    println!("{}", random_matrix);

    //  let normalised = normalise(random_matrix.clone());
    //  println!("normalised {}", normalised.clone());
    //
    //  let mean_centered = center_at_mean(normalised);
    //  println!("mean centered {}", mean_centered.clone());

    let cov_matrix = covariance_matrix(random_matrix.clone()).unwrap();
    println!("cov matrix: {} ", cov_matrix);

    eigendecomposition(cov_matrix);
    //let xs = vec![1.0, 3.0, 6.0, 7.0];
    //let ys = vec![2.0, 5.0, 2.0, 3.0];
    //println!("cov: {} ", population_covariance(&xs, &ys).unwrap());
}
