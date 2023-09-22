use nalgebra::{vector, DMatrix, DVector, Dyn, Dynamic, OMatrix, Vector3};

// Alias for Dynamically sized and dynamically allocated matrix.
type MatrixF64 = OMatrix<f64, Dyn, Dyn>;

/// Rows are observations, columns are variables/features.
pub fn rand_matrix() -> MatrixF64 {
    DMatrix::new_random(3, 3)
}

pub fn normalise(mut m: OMatrix<f64, Dyn, Dyn>) -> MatrixF64 {
    for mut col in m.column_iter_mut() {
        let normalized = col.normalize(); // Help WHat does this actually do?
                                          // What does it mean to normalise data?
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
    let mut col_means = Vec::<f64>::new();

    for i in 0..m.ncols() {
        let column: DVector<f64> = m.column(i).into();
        let column_sum: f64 = column.data.as_vec().iter().sum();
        let column_mean = column_sum / num_rows as f64;
        col_means.push(column_mean);
    }

    let means_row: DVector<f64> = DVector::from_vec(col_means.clone());

    let centering_matrix: OMatrix<f64, Dyn, Dyn> =
        OMatrix::<f64, Dyn, Dyn>::from_fn(num_rows, m.ncols(), |_r, c| col_means[c]);
    println!("cen {}", centering_matrix.clone());

    let mean_centered = m.clone() - centering_matrix;
    mean_centered
}

/// Gets the cov matrix (C) of rows (R):
/// C = R^T * R
/// ---- Covariance ----
/// Measures the direction of the linear relationship between two variables
/// extending the notion of variance of one dimension.
///
/// Bigger positive values mean x, y are increasing together
/// Smaller negative values mean x, y are decreasing together.
///
/// Cov(x,y) = 1/(n-1)(sum[(x_i - x_bar])*(y_i - y_bar))
///
/// Note covariance doesn't tell us anything about the strength
/// of the relationship. For example just because x,y are increasing
/// together it may be that a big jump in x
/// only results in a tiny jump in y.
///
/// ---- Covariance Matrix ----
/// For two dimensions would look like:
///  | Cov(x,x)  Cov(x,y) |
///  | Cov(x,y)  Cov(y,y) |
///
/// Diagonals represent variance as Cov(y,y) = Var(y) (variance of y)
/// Non-diagonals represent covariance.
///
/// Cov can be anything from -inf to +inf
/// Also remember covariance is sensitive to scaling unlike
/// the correlation coefficient.
fn get_covariance_matrix(m: OMatrix<f64, Dyn, Dyn>) {
    todo!()
}

// V_1^T * R^T * R * v1
fn _get_eigens_of_cov_matrix() {}

fn _eigendecomp_of_cov_matrix() {}

pub fn _get_principal_comps() {
    // First mean centre each row in the matrix.
}
