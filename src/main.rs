use nalgebra::DMatrix;

use pca::pca;

fn main() {
    // columns are features, rows are observations.
    let rand_3x3_matrix = DMatrix::new_random(3, 3);

    let transformed_data = pca(rand_3x3_matrix);

    println!("transformed data: {} ", transformed_data);
}
