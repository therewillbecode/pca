use anyhow::anyhow;
use core::cmp::Ordering;
use nalgebra::{DMatrix, DVector};

pub fn pca(m: DMatrix<f64>) -> DMatrix<f64> {
    let normalised_data = normalise(m);

    let cov_matrix = covariance_matrix(normalised_data.clone()).unwrap();

    let principal_comps = principal_components(cov_matrix).unwrap();

    transform_data(normalised_data, principal_comps)
}

// Gets the mean of each column of the matrix as a vector.
fn column_means(m: &DMatrix<f64>) -> DVector<f64> {
    let mut col_means = Vec::<f64>::new(); // The means of our matrix columns.
    for i in 0..m.ncols() {
        let column: DVector<f64> = m.column(i).into();
        let column_sum: f64 = column.data.as_vec().iter().sum();
        let column_mean = column_sum / m.nrows() as f64;
        col_means.push(column_mean);
    }

    DVector::from_vec(col_means.clone())
}

fn column_stddevs(m: &DMatrix<f64>) -> DVector<f64> {
    let mut col_stddevs = Vec::<f64>::new();
    for i in 0..m.ncols() {
        let column: DVector<f64> = m.column(i).into();
        let column_sum: f64 = column.data.as_vec().iter().sum();
        let column_mean = column_sum / m.nrows() as f64;

        col_stddevs.push(std_deviation(column_mean, column.data.as_vec()));
    }

    DVector::from_vec(col_stddevs.clone())
}

fn std_deviation(mean: f64, data: &Vec<f64>) -> f64 {
    let variance = data
        .iter()
        .map(|value| {
            let diff = mean - (*value as f64);

            diff * diff
        })
        .sum::<f64>()
        / (data.len() - 1) as f64;

    variance.sqrt()
}

// AKA column centering (columns are variables).
// We just move the origin of the coordinate system to coincide
// with the average value for each dimension.
fn normalise(m: DMatrix<f64>) -> DMatrix<f64> {
    let means_row: DVector<f64> = column_means(&m);
    let centering_matrix: DMatrix<f64> =
        DMatrix::<f64>::from_fn(m.nrows(), m.ncols(), |_r, c| means_row[c]);
    let demeaned = m.clone() - centering_matrix.clone();

    let std_devs: DVector<f64> = column_stddevs(&m);
    DMatrix::<f64>::from_fn(m.nrows(), m.ncols(), |r, c| {
        demeaned.column(c)[r] / std_devs[c]
    })
}

/// think of covariance as average product of distances from means
// which gives the direction of the relationship but not strength.
fn population_covariance(xs: &Vec<f64>, ys: &Vec<f64>) -> anyhow::Result<f64> {
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

    let covariance = xs
        .iter()
        .enumerate()
        .map(|(pos, _i)| (xs[pos] - mean_xs) * (ys[pos] - mean_ys))
        .sum::<f64>()
        / (xs.len()) as f64;

    Ok(covariance)
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
fn covariance_matrix(m: DMatrix<f64>) -> anyhow::Result<DMatrix<f64>> {
    if m.ncols() < 2 || m.nrows() < 2 {
        return Err(anyhow!(
            "Cannot compute covariance matrix unless rows > 1 and cols > 1"
        ));
    }

    let cov_matrix: DMatrix<f64> = DMatrix::from_fn(m.ncols(), m.ncols(), |r, c| {
        let xs: DVector<f64> = m.column(r).into();
        let ys: DVector<f64> = m.column(c).into();

        population_covariance(ys.data.as_vec(), xs.data.as_vec()).unwrap() // fixme
    });

    Ok(cov_matrix)
}

#[derive(Debug, Clone)]
struct Eig {
    eigenvector: Vec<f64>,
    eigenvalue: f64,
    explained_var: f64,
}

/// The eigendecomposition is used because we want to figure out
/// which directions explain the most variance and by how much.
/// The eigenvector with the biggest eigenvalue is the direction which
/// explains the most variance. Eigenvalues near 0 means we may discard these
/// components later potentially.
fn principal_components(m: DMatrix<f64>) -> anyhow::Result<Vec<Eig>> {
    // get the eigenvectors and eigenvalues of the covariance matrix.
    let eigenvalues: DVector<f64> = m.clone().symmetric_eigen().eigenvalues.column(0).into();
    let sum_eigenvalues: f64 = eigenvalues.data.as_vec().to_owned().iter().sum();

    // each column is an eigenvector
    let eigenvectors: DMatrix<f64> = m.clone().symmetric_eigen().eigenvectors;

    let mut eigs: Vec<Eig> = (0..m.ncols())
        .map(|col_ix| {
            let v: DVector<f64> = eigenvectors.column(col_ix).to_owned().into();

            Eig {
                eigenvector: v.data.as_vec().to_owned(),
                eigenvalue: eigenvalues[col_ix],
                explained_var: eigenvalues[col_ix] / sum_eigenvalues,
            }
        })
        .collect::<Vec<Eig>>();

    eigs.sort_by(|a, b| {
        if a.eigenvalue > b.eigenvalue {
            Ordering::Less
        } else if a.eigenvalue == b.eigenvalue {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    });

    Ok(eigs)
}

// The resulting projected data are essentially linear combinations
// of the original data capturing most of the variance in the data
fn transform_data(data: DMatrix<f64>, eigs_sorted: Vec<Eig>) -> DMatrix<f64> {
    let eigenvectors: DMatrix<f64> =
        DMatrix::<f64>::from_fn(eigs_sorted.len(), eigs_sorted.len(), |r, c| {
            eigs_sorted[c].eigenvector[r]
        });

    // apply the change of basis
    data * eigenvectors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_1() {
        let xs = vec![1.0, 3.0, 6.0, 7.0];
        let ys = vec![2.0, 5.0, 2.0, 3.0];
        let res = population_covariance(&xs, &ys).unwrap();

        assert_eq!(res, -0.25);
    }

    #[test]
    fn covariance_2() {
        let xs = vec![1.0, -99.0, 3.0, 4.0, 1.0];
        let ys = vec![1.0, 2.0, 3.0, 0.0, 0.8];
        let res = population_covariance(&xs, &ys).unwrap();

        assert_eq!(res, -12.959999999999999);
    }

    #[test]
    fn covariance_3() {
        let xs = vec![1.0, 2.0];
        let ys = vec![3.0, 4.0];
        let res = population_covariance(&xs, &ys).unwrap();
        assert_eq!(res, 0.25);
    }

    #[test]
    fn covariance_empty_should_fail() {
        let xs = vec![];
        let ys = vec![];
        let res = population_covariance(&xs, &ys);

        assert!(res.is_err());
    }

    #[test]
    fn covariance_matrix_fails_when_less_than_smaller_than_2x2() {
        let m = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        let res = covariance_matrix(m);
        assert!(res.is_err());
    }

    #[test]
    fn covariance_matrix_computes_2x2() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let res = covariance_matrix(m).unwrap().data.as_vec().to_owned();
        let expected: Vec<f64> = vec![1.0; 2 * 2];

        assert_eq!(res, expected);
    }

    #[test]
    fn covariance_matrix_computes_3x3() {
        let m = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let res = covariance_matrix(m).unwrap().data.as_vec().to_owned();
        let expected: Vec<f64> = vec![6.0; 3 * 3];

        assert_eq!(res, expected);
    }
}
