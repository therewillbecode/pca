use anyhow::anyhow;
use nalgebra::{DMatrix, DVector};

/// Rows are observations, columns are variables/features.
pub fn rand_matrix() -> DMatrix<f64> {
    DMatrix::new_random(3, 3)
}

pub fn normalise(mut m: DMatrix<f64>) -> DMatrix<f64> {
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
pub fn center_at_mean(m: DMatrix<f64>) -> DMatrix<f64> {
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

    let centering_matrix: DMatrix<f64> =
        DMatrix::<f64>::from_fn(num_rows, m.ncols(), |_r, c| col_means[c]);

    let mean_centered = m.clone() - centering_matrix;
    mean_centered
}

/// think of covariance as average product of distances from means
// which gives the direction of the relationship but not strength.
pub fn population_covariance(xs: Vec<f64>, ys: Vec<f64>) -> anyhow::Result<f64> {
    if !(xs.len() == ys.len()) {
        return Err(anyhow!(
            "Cannot compute covariance of vec of different lengths"
        ));
    };

    if xs.is_empty() {
        return Err(anyhow!("Cannot compute covariance of empty vec"));
    };

    let mean_xs: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
    let mean_ys: f64 = ys.iter().sum::<f64>() / ys.len() as f64;

    Ok(xs
        .iter()
        .enumerate()
        .map(|(pos, i)| (xs[pos] - mean_xs) * (ys[pos] - mean_ys))
        .sum::<f64>()
        / (xs.len()) as f64)
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
/// For 3 dimensions would look like:
///  | Cov(x,x)  Cov(x,y) Cov(x,z) |
///  | Cov(x,y)  Cov(y,y) Cov(y,z) |
///  | Cov(x,z)  Cov(y,z) Cov(z,z) |
///
/// Diagonals represent variance as Cov(y,y) = Var(y) since
//  covariance of something with itself is just variance
//
/// Non-diagonals represent covariance.
///
/// Cov can be anything from -inf to +inf
/// Also remember covariance is sensitive to scaling unlike
/// the correlation coefficient.
fn get_covariance_matrix(m: DMatrix<f64>) {
    let covariances = todo!();

    let cov_matrix: DMatrix<f64> = DMatrix::from_vec(m.ncols(), m.ncols(), covariances);
}

// V_1^T * R^T * R * v1
fn _get_eigens_of_cov_matrix() {}

fn _eigendecomp_of_cov_matrix() {}

pub fn _get_principal_comps() {
    // First mean centre each row in the matrix.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance() {
        let xs = vec![1.0, 3.0, 6.0, 7.0];
        let ys = vec![2.0, 5.0, 2.0, 3.0];
        let res = population_covariance(xs, ys).unwrap();

        assert_eq!(res, -0.25);
    }

    #[test]
    fn covariance_empty_should_fail() {
        let xs = vec![];
        let ys = vec![];
        let res = population_covariance(xs, ys);

        assert!(res.is_err());
    }
}
