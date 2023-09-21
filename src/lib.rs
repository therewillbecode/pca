use nalgebra::{vector, DMatrix, Dyn, OMatrix, Vector3};

// Dynamically sized and dynamically allocated matrix with
// two rows and using 32-bit signed integers.
type DMatrixf32 = OMatrix<f32, Dyn, Dyn>;

pub fn rand_matrix() -> DMatrixf32 {
    DMatrix::new_random(4, 4)
}
