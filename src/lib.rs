use anyhow::anyhow;
use core::cmp::Ordering;
use nalgebra::linalg::SymmetricEigen;
use nalgebra::{DMatrix, DVector};

/// Rows are observations, columns are variables/features.
pub fn rand_matrix() -> DMatrix<f64> {
    //  DMatrix::new_random(3, 3)
    DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
}

pub fn normalise(mut m: DMatrix<f64>) -> DMatrix<f64> {
    for mut col in m.column_iter_mut() {
        let normalized = col.normalize(); // Help WHat does this actually do?
                                          // What does it mean to normalise data?
        col.copy_from(&normalized);
    }

    m
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

// AKA column centering (columns are variables).
// We just move the origin of the coordinate system to coincide
// with the average value for each dimension.
pub fn center_at_mean(m: DMatrix<f64>) -> DMatrix<f64> {
    let means_row: DVector<f64> = column_means(&m);

    let centering_matrix: DMatrix<f64> =
        DMatrix::<f64>::from_fn(m.nrows(), m.ncols(), |_r, c| means_row[c]);

    m - centering_matrix
}

/// think of covariance as average product of distances from means
// which gives the direction of the relationship but not strength.
pub fn population_covariance(xs: &Vec<f64>, ys: &Vec<f64>) -> anyhow::Result<f64> {
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
        .map(|(pos, i)| (xs[pos] - mean_xs) * (ys[pos] - mean_ys))
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
pub fn covariance_matrix(m: DMatrix<f64>) -> anyhow::Result<DMatrix<f64>> {
    if m.ncols() < 2 || m.nrows() < 2 {
        return Err(anyhow!(
            "Cannot compute covariance matrix unless rows > 1 and cols > 1"
        ));
    }

    // TODO1  - need to special case when number of rows = 1 and columns = 2 so that the returned matrix is just the covar(a,b)
    // todo 2 also need to ensure min of 2 columns exists
    let cov_matrix: DMatrix<f64> = DMatrix::from_fn(m.ncols(), m.ncols(), |r, c| {
        let xs: DVector<f64> = m.column(r).into();
        let ys: DVector<f64> = m.column(c).into();

        population_covariance(ys.data.as_vec(), xs.data.as_vec()).unwrap() // fixme
    });

    Ok(cov_matrix)
}

pub fn eigendecomposition_of_cov_matrix() {
    // Since matrices encode linear transformations then our covariance matrix A
    // can be seen as a kind of linear transformation and will have its own
    // characteristic polynomial (f). The roots of which give use the eigenvalues
    // of A.
    //
    // f(λ) = det(A−λI)
    //
    // At the roots of f we know linear transformation represented by the matrix A
    // minus the scaling factor λ has a determinant of 0 which means the directional
    // part of the linear transformation is 0 (no direction change)

    // Once we have the eigenvalues we can find the eigenvectors.
    // We just set our λ to the roots of the characteristic polynomial and then
    // solve the resulting system of linear equations.
}

#[derive(Debug, Clone)]
pub struct Eig {
    pub eigenvector: Vec<f64>,
    pub eigenvalue: f64,
    pub explained_var: f64,
}

/// The eigendecomposition is used because we want to figure out
/// which directions explain the most variance and by how much.
/// The eigenvector with the biggest eigenvalue is the direction which
/// explains the most variance. Eigenvalues near 0 means we may discard these
/// components later potentially.
pub fn principal_components(m: DMatrix<f64>) -> anyhow::Result<Vec<Eig>> {
    // get the eigenvectors and eigenvalues of the covariance matrix.
    let eigenvalues: DVector<f64> = m
        .clone()
        .symmetric_eigen()
        .eigenvalues
        //.ok_or_else(|| anyhow!("No eigenvalues found"))
        .column(0)
        .into();
    let mut sum_eigenvalues: f64 = eigenvalues.data.as_vec().to_owned().iter().sum();

    println!("eignvals, {}", eigenvalues);

    // each column is an eigenvector
    let eigenvectors: DMatrix<f64> = m.clone().symmetric_eigen().eigenvectors;

    println!("eignvector, {}", eigenvectors.clone());

    let mut eigs: Vec<Eig> = ((0 as usize)..(m.ncols() as usize))
        .into_iter()
        .map(|col_ix| {
            let v: DVector<f64> = eigenvectors.column(col_ix).to_owned().into();

            Eig {
                eigenvector: v.data.as_vec().to_owned(),
                eigenvalue: eigenvalues[col_ix],
                explained_var:  eigenvalues[col_ix]/ sum_eigenvalues
            }
        })
        .collect::<Vec<Eig>>();

        println!("ctor, {:?}", eigs.clone());

    eigs.sort_by(|a, b| {
        if a.eigenvalue > b.eigenvalue {
            Ordering::Less
        } else if a.eigenvalue == b.eigenvalue {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    });

   // print!("eigs {:?}", eigs.clone());

    Ok(eigs)
}

// The resulting projected data are essentially linear combinations
// of the original data capturing most of the variance in the data
pub fn transform_data(data: DMatrix<f64>, eigs_sorted: Vec<Eig>) -> DMatrix<f64> {
    let eigenvectors: DMatrix<f64> =
        DMatrix::<f64>::from_fn(eigs_sorted.len(),eigs_sorted.len(),
         |r, c| eigs_sorted[c].eigenvector[r]).transpose();

         println!("esssssssssssssseee {:?}", eigenvectors);
    // take the transpose

    // multiply eigs^T * original_data

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

    #[test]
    fn eigen_of_3x3() {
        let m = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let eigenvectors = m
            .clone()
            .symmetric_eigen()
            .eigenvectors
            .data
            .as_vec()
            .to_owned();
        let expected_eigenvectors: Vec<f64> = vec![
            0.40962667186957685,
            0.5426486477297079,
            0.7333065080920609,
            -0.802388907893736,
            -0.16812656383796679,
            0.5726303336543874,
            -0.43402537965210164,
            0.8229616659657703,
            -0.3665461310240397,
        ];
        assert_eq!(eigenvectors, expected_eigenvectors);

        let eigenvalues = m.symmetric_eigen().eigenvalues.data.as_vec().to_owned();
        let expected_eigenvalues: Vec<f64> =
            vec![18.830235795506823, -3.1574678406080756, -0.6727679548987634];
        assert_eq!(eigenvalues, expected_eigenvalues);
    }

    #[test]
    fn eigen_of_4x4() {
        let m = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let eigenvectors = m
            .clone()
            .symmetric_eigen()
            .eigenvectors
            .data
            .as_vec()
            .to_owned();
        let expected_eigenvectors: Vec<f64> = vec![
            0.35441037468080533,
            0.4227352047406468,
            0.5198236679927599,
            0.6522818311022068,
            -0.7270046361666769,
            -0.3847562596537509,
            0.10790023406162648,
            0.5583765925778588,
            -0.5222356777307557,
            0.4766781850093164,
            0.5405251874455946,
            -0.4559389504947205,
            -0.2704208612808326,
            0.6678588737647819,
            -0.6526663890694112,
            0.23422994491923008,
        ];
        assert_eq!(eigenvectors, expected_eigenvectors);

        let eigenvalues = m.symmetric_eigen().eigenvalues.data.as_vec().to_owned();
        let expected_eigenvalues: Vec<f64> = vec![
            44.09059195493653,
            -7.674245249726649,
            -1.5293393614263537,
            -0.8870073437835344,
        ];
        assert_eq!(eigenvalues, expected_eigenvalues);
    }
}
