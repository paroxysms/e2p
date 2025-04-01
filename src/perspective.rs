use ndarray::{Axis, s, Zip, concatenate, stack};
use ndarray_linalg::Inverse;
use opencv::{imgcodecs, prelude};
use opencv::calib3d::rodrigues;
use opencv::prelude::{MatExprTraitConst, MatTraitConst, MatTraitConstManual};

pub struct Equirectangular {
    src: prelude::Mat,
    height: i32,
    width: i32,
}

impl Equirectangular {
    pub fn new(img_name: &str) -> Equirectangular {
        let src = imgcodecs::imread(img_name, imgcodecs::IMREAD_COLOR).expect("Could not read image!");
        let height = src.cols();
        let width = src.rows();

        Equirectangular {
            src,
            height,
            width,
        }
    }

    pub fn get_perspective(&self, fov: f64, theta: f64, phi: f64, height: u32, width: u32) -> prelude::Mat {
        let f = 0.5 * (width as f64) * 1.0 / f64::tan(0.5 * fov / 180.0 * std::f64::consts::PI);
        let cx = (width as f64 - 1.0) / 2.0;
        let cy = (height as f64 - 1.0) / 2.0;
        let k: ndarray::Array2<f64> = ndarray::arr2(&[
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0],
        ]);

        let k_inv = k.inv().expect("Could not invert matrix!");

        let x = ndarray::Array2::from_shape_fn((height as usize, width as usize), |(_i, j)| j as f64);
        let y = ndarray::Array2::from_shape_fn((height as usize, width as usize), |(i, _)| i as f64);
        let z = ndarray::Array2::from_elem((height as usize, width as usize), 1.0);
        let xyz = stack(Axis(2), &[x.view(), y.view(), z.view()])
            .expect("Failed to stack arrays");

        let n_points = (height as usize) * (width as usize);
        let xyz_2d = xyz.to_shape((n_points, 3)).expect("Failed to reshape xyz").to_owned();
        let transformed = xyz_2d.dot(&k_inv.t());
        let reshaped_xyz = transformed.to_shape((height as usize, width as usize, 3)).expect("Failed to reshape transformed xyz").to_owned();

        let y_axis = opencv::core::Vec3d::from([0.0, 1.0, 0.0]);
        let x_axis = opencv::core::Vec3d::from([1.0, 0.0, 0.0]);

        let theta_rad = theta.to_radians();
        let phi_rad = phi.to_radians();

        let mut r1 = prelude::Mat::default();
        rodrigues(&opencv::core::Mat::from_slice(&[y_axis[0] * theta_rad, y_axis[1] * theta_rad, y_axis[2] * theta_rad]).unwrap(), &mut r1, &mut opencv::core::Mat::default()).unwrap();

        let mut r2 = prelude::Mat::default();
        let x_axis_mat = opencv::core::Mat::from_slice_2d(&[[x_axis[0]], [x_axis[1]], [x_axis[2]]]).unwrap();
        let mut r1_dot_x_axis = prelude::Mat::default();
        opencv::core::gemm(&r1, &x_axis_mat, 1.0, &prelude::Mat::default(), 0.0, &mut r1_dot_x_axis, 0).unwrap();
        let r1_dot_x_axis_vec = r1_dot_x_axis.to_vec_2d::<f64>().unwrap();
        rodrigues(&opencv::core::Mat::from_slice(&[
            r1_dot_x_axis_vec[0][0] * phi_rad,
            r1_dot_x_axis_vec[1][0] * phi_rad,
            r1_dot_x_axis_vec[2][0] * phi_rad
        ]).unwrap(), &mut r2, &mut opencv::core::Mat::default()).unwrap();

        let r = (r2 * r1).into_result().unwrap().to_mat().unwrap();

        let r_vec = r.to_vec_2d::<f64>().unwrap(); // Vec<Vec<f64>>
        let r_nd = ndarray::Array2::from_shape_vec((3, 3), r_vec.into_iter().flatten().collect())
            .expect("Failed to create ndarray from r");

        let reshaped_xyz_2d = reshaped_xyz.to_shape((n_points, 3)).expect("Failed to reshape reshaped_xyz").to_owned();
        let rotated = reshaped_xyz_2d.dot(&r_nd.t());
        let rotated_xyz = rotated.to_shape((height as usize, width as usize, 3))
            .expect("Failed to reshape rotated xyz").to_owned();

        let lonlat = xyz_to_lonlat(rotated_xyz);
        let xy = lonlat_to_xy(lonlat, (self.width as usize, self.height as usize));

        let mut persp = prelude::Mat::default();

        let binding = xy.map_axis(Axis(2), |v| v[0] as f32)
            .into_dimensionality::<ndarray::Ix2>().unwrap();
        let x_values = binding
            .as_standard_layout();
        let binding = xy.map_axis(Axis(2), |v| v[1] as f32)
            .into_dimensionality::<ndarray::Ix2>().unwrap();
        let y_values = binding
            .as_standard_layout();
        let (r_rows, r_cols) = x_values.dim();
        let x = prelude::Mat::new_rows_cols_with_data(r_rows as i32, r_cols as i32, x_values.as_slice().unwrap()).unwrap();
        let y = prelude::Mat::new_rows_cols_with_data(r_rows as i32, r_cols as i32, y_values.as_slice().unwrap()).unwrap();

        opencv::imgproc::remap(
            &self.src, &mut persp,
            &x,
            &y,
            opencv::imgproc::INTER_CUBIC,
            opencv::core::BORDER_WRAP,
            opencv::core::Scalar::all(0.0)
        ).unwrap();

        persp
    }
}

fn xyz_to_lonlat(xyz: ndarray::Array3<f64>) -> ndarray::Array3<f64> {
    let norm = xyz.map_axis(Axis(2), |v| v.dot(&v).sqrt())
        .insert_axis(Axis(2));
    let xyz_norm = xyz / norm;

    let x = xyz_norm.slice(s![.., .., 0..1]).to_owned();
    let y = xyz_norm.slice(s![.., .., 1..2]).to_owned();
    let z = xyz_norm.slice(s![.., .., 2..3]).to_owned();

    let lon = Zip::from(x.view())
        .and(z.view())
        .map_collect(|&a, &b| a.atan2(b));
    let lat = y.mapv(|a| a.asin());

    concatenate(Axis(2), &[lon.view(), lat.view()]).expect("Failed to concatenate the arrays")
}

fn lonlat_to_xy(lonlat: ndarray::Array3<f64>, shape: (usize, usize)) -> ndarray::Array3<f64> {
    let (h, w) = shape;
    let x = lonlat
        .slice(s![.., .., 0..1])
        .to_owned()
        .mapv(|v| (v / (2.0 * std::f64::consts::PI) + 0.5) * ((w as f64) - 1.0));
    let y = lonlat
        .slice(s![.., .., 1..2])
        .to_owned()
        .mapv(|v| (v / std::f64::consts::PI + 0.5) * ((h as f64) - 1.0));
    concatenate(Axis(2), &[x.view(), y.view()])
        .expect("Failed to concatenate X and Y")
}