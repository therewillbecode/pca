use nalgebra::{vector, DMatrix, DVector, Dyn, Dynamic, OMatrix, Vector3};

// Dynamically sized and dynamically allocated matrix with
type MatrixF64 = OMatrix<f64, Dyn, Dyn>;

/// Rows are observations, columns are variables.
pub fn rand_matrix() -> MatrixF64 {
    DMatrix::new_random(3, 3)
}

pub fn normalise(mut m: OMatrix<f64, Dyn, Dyn>) -> MatrixF64 {
    for mut col in m.column_iter_mut() {
        let normalized = col.normalize();
        col.copy_from(&normalized);
    }

    m
}

// AKA column centering (columns are variables).
// We just move the origin of the coordinate system to coincide
// with the average value for each dimension.
pub fn center_at_mean(m: OMatrix<f64, Dyn, Dyn>) -> MatrixF64 {
    // Get columns
    //  let rows = m.columns_generic_mut(0, m.ncols());

    let num_rows = m.nrows();
    // Get "mean row" which is vector where each element is the mean
    // of its column.
    let mut means = Vec::<f64>::new();

    for i in 0..m.ncols() {
        let column: DVector<f64> = m.column(i).into();
        let column_sum: f64 = column.data.as_vec().iter().sum();
        let column_mean = column_sum / num_rows as f64;
        means.push(column_mean);
        println!("mean {}", column_mean);
    }

    let means_row: DVector<f64> = DVector::from_vec(means);

    //    Builds a vector filled with a constant.
    //let vector_of_ones: DVector<f64> = DVec::from_elem(m.ncols(), 1.0);

    // Multiply by the centering matrix

    // Initialise Column length vector with mean of R for each R.

    // Then mean centre each row in the matrix.
    // m.new_translation(&)
    m
}

// We should also scale variables since our features may have different units.

// Gets the cov matrix (C) of rows (R):
// C = R^T * R
fn get_covariance_matrix() {
    // ---- Covariance ----
    // Measures the direction of the linear relationship between two variables
    // extending the notion of variance of one dimension.
    //
    // Bigger positive values mean x, y are increasing together
    // Smaller negative values mean x, y are decreasing together.
    //
    // Cov(x,y) = 1/(n-1)(sum[(x_i - x_bar])*(y_i - y_bar))
    //
    // Note covariance doesn't tell us anything about the strength
    // of the relationship. For example just because x,y are increasing
    // together it may be that a big jump in x
    // only results in a tiny jump in y.
    //
    // ---- Covariance Matrix ----
    // For two dimensions would look like:
    //  | Cov(x,x)  Cov(x,y) |
    //  | Cov(x,y)  Cov(y,y) |
    //
    // Diagonals represent variance as Cov(y,y) = Var(y) (variance of y)
    // Non-diagonals represent covariance.
    //
    // Cov can be anything from -inf to +inf
    // Also remember covariance is sensitive to scaling unlike
    // the correlation coefficient.
}

// V_1^T * R^T * R * v1
fn get_eigens_of_cov_matrix() {}

fn eigendecomp_of_cov_matrix() {}

pub fn get_principal_comps() {
    // First mean centre each row in the matrix.
}
