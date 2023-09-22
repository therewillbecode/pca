mod lib;

use pca::*;

fn main() {
    let mut random_matrix = lib::rand_matrix();
    println!("{}", random_matrix);

    let normalised = normalise(random_matrix.clone());
    println!("normalised {}", normalised.clone());

    let mean_centered = center_at_mean(normalised);
    println!("mean centered {}", mean_centered.clone());
}
