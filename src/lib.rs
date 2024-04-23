use pyo3::prelude::*;
mod network;
use crate::network::network::Network;
use nalgebra::DMatrix;
use ndarray::ArrayViewMut2;
use ndarray::{Array, Array1, Array2, Array3, Axis};
use ndarray_stats::QuantileExt;
use rand::distributions::Uniform;
use rand::Rng;
use sphrs::{Coordinates, HarmonicsSet, RealSH};
use std::fs::File;
use std::io::Read;
use std::time::Instant;

const VDF_SIZE: usize = 50;
const MIN_VAL: f64 = 1e-16;

//Magic numbers
const DEG2RAD: f32 = std::f32::consts::PI / 180.0;
const NSHELLS: usize = 50;
const NTHETA: usize = 180;
const NPHI: usize = 360;
const DTHETA: f32 = 1.0;
const DPHI: f32 = 1.0;
//~Magic numbers

// ******************* MLP COMPRESSION *************************************//
fn read_vdf_from_file(filename: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    println!("Reading VDF from {filename}");
    let mut f = File::open(filename)?;
    let mut buffer: Vec<u8> = vec![];
    f.read_to_end(&mut buffer)?;
    let mut vdf: Vec<f64> = vec![];
    let mut slice: [u8; 8] = [0; 8];
    for i in (0..buffer.len()).step_by(8) {
        for j in 0..8 {
            slice[j] = buffer[i + j];
        }
        vdf.push(f64::from_ne_bytes(slice));
    }
    assert_eq!(vdf.len(), VDF_SIZE * VDF_SIZE * VDF_SIZE);
    Ok(vdf)
}

fn vdf_fourier_features(vdf: &Vec<f64>, order: usize) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    let total_dims = 3 + order * 6;

    let mut harmonics: Vec<f64> = vec![0.0; order];
    harmonics.resize(order, 0.0);
    let mut rng = rand::thread_rng();
    let range = Uniform::<f64>::new(-Into::<f64>::into(0.0), Into::<f64>::into(6.0));
    harmonics.iter_mut().for_each(|x| *x = rng.sample(&range));

    let mut vspace = DMatrix::<f64>::zeros(vdf.len(), total_dims);
    let mut density = DMatrix::<f64>::zeros(vdf.len(), 1);

    // Iterate over pixels
    let mut counter = 0;
    for z in 0..VDF_SIZE {
        for y in 0..VDF_SIZE {
            for x in 0..VDF_SIZE {
                let pos_z = z as f64 / VDF_SIZE as f64 - 0.5;
                let pos_y = y as f64 / VDF_SIZE as f64 - 0.5;
                let pos_x = x as f64 / VDF_SIZE as f64 - 0.5;
                vspace[(counter, 0)] = pos_x;
                vspace[(counter, 1)] = pos_y;
                vspace[(counter, 2)] = pos_z;
                for f in 0..order {
                    vspace[(counter, f * 6 + 3)] =
                        (harmonics[f] * 2.0 * std::f64::consts::PI * pos_x).sin();
                    vspace[(counter, f * 6 + 4)] =
                        (harmonics[f] * 2.0 * std::f64::consts::PI * pos_y).sin();
                    vspace[(counter, f * 6 + 5)] =
                        (harmonics[f] * 2.0 * std::f64::consts::PI * pos_z).sin();
                    vspace[(counter, f * 6 + 6)] =
                        (harmonics[f] * 2.0 * std::f64::consts::PI * pos_x).cos();
                    vspace[(counter, f * 6 + 7)] =
                        (harmonics[f] * 2.0 * std::f64::consts::PI * pos_y).cos();
                    vspace[(counter, f * 6 + 8)] =
                        (harmonics[f] * 2.0 * std::f64::consts::PI * pos_z).cos();
                }
                density[(counter, 0)] = vdf[counter];
                counter += 1;
            }
        }
    }
    (vspace, density, harmonics)
}

fn scale_vdf(vdf: &mut Vec<f64>) {
    vdf.iter_mut()
        .for_each(|x| *x = f64::abs(f64::log10(f64::max(*x, MIN_VAL))));
}

fn unscale_vdf(vdf: &mut Vec<f64>) {
    vdf.iter_mut()
        .for_each(|x| *x = f64::max(f64::powf(10.0, -1.0 * *x), MIN_VAL));
}

fn reconstruct_vdf(net: &mut Network<f64>, vspace: &DMatrix<f64>) -> Vec<f64> {
    let mut sample = DMatrix::<f64>::zeros(1, vspace.ncols());
    let mut buffer = DMatrix::<f64>::zeros(1, 1);
    let mut reconstructed_vdf: Vec<f64> = vec![];
    let mut decoder = Network::<f64>::new_from_other_with_batchsize(&net, 1);
    for s in 0..vspace.nrows() {
        for i in 0..vspace.ncols() {
            sample[(0, i)] = vspace[(s, i)];
        }
        decoder.eval(&sample, &mut buffer);
        reconstructed_vdf.push(buffer[(0, 0)]);
    }
    reconstructed_vdf
}

fn compress_vdf(
    vdf: &Vec<f64>,
    fourier_order: usize,
    epochs: usize,
    n_layers: usize,
    n_neurons: usize,
) -> Vec<f64> {
    let (vspace, density, _harmonics) = vdf_fourier_features(&vdf, fourier_order);
    let mut net = Network::<f64>::new(
        vspace.ncols(),
        density.ncols(),
        n_layers,
        n_neurons,
        &vspace,
        &density,
        8,
    );
    //Train
    let before = Instant::now();
    for epoch in 0..epochs {
        let cost = net.train_minibatch(1e-3.into(), epoch, 1);
        if epoch % 1 == 0 {
            println!("Cost at epoch {} is {:.4}", epoch, cost);
        }
    }
    let t_spent = before.elapsed();
    let reconstructed = reconstruct_vdf(&mut net, &vspace);
    let vdf_size = vdf.len() * std::mem::size_of::<f64>();
    let net_size = net.calculate_total_bytes();
    let compression_ratio = vdf_size as f32 / net_size as f32;
    let _bytes = net.get_state();
    println!(
        "Done in {:.2} s.  Compression ratio = {:.2}x .",
        t_spent.as_secs_f32(),
        compression_ratio
    );
    reconstructed
}

//~ ******************* MLP COMPRESSION *************************************//

// ******************* SPH COMPRESSION *************************************//
/*
 Kostis Papadakis,2024 (kpapadakis@protonmail.com)
 Some very beatiful math in here : https://basesandframes.wordpress.com/wp-content/uploads/2016/05/spherical_harmonic_lighting_gritty_details_green_2003.pdf

 ***** WARNING ****

 This is  a prototype that only works with double precission VDFs on a 50x50x50 uniform Cartesian grid.
 Do not use it if your situation differs!
 Could it be better? Yes. Is it. No.

 ******************
*/

fn read_vdf_from_file_to_nd(filename: &str) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
    println!("Reading VDF from {filename}");
    let mut f = File::open(filename)?;
    let mut buffer: Vec<u8> = vec![];
    f.read_to_end(&mut buffer)?;
    let mut vdf_buffer: Vec<f64> = vec![];
    let mut slice: [u8; 8] = [0; 8];
    for i in (0..buffer.len()).step_by(8) {
        for j in 0..8 {
            slice[j] = buffer[i + j];
        }
        vdf_buffer.push(f64::from_ne_bytes(slice));
    }

    let mut vdf: Array3<f64> = Array3::<f64>::zeros((VDF_SIZE, VDF_SIZE, VDF_SIZE));
    let mut index = 0;
    for k in 0..VDF_SIZE {
        for j in 0..VDF_SIZE {
            for i in 0..VDF_SIZE {
                vdf[[i, j, k]] = vdf_buffer[index];
                index += 1;
            }
        }
    }
    assert_eq!(vdf.len(), VDF_SIZE * VDF_SIZE * VDF_SIZE);
    Ok(vdf)
}

fn get_linspace(start: f64, stop: f64, num: usize) -> Array1<f64> {
    Array::linspace(start, stop, num)
}

fn meshgrid(
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let nx = x.len();
    let ny = y.len();
    let nz = z.len();

    let mut xx = Array3::<f64>::zeros((nx, ny, nz));
    let mut yy = Array3::<f64>::zeros((nx, ny, nz));
    let mut zz = Array3::<f64>::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                xx[[i, j, k]] = x[i];
                yy[[i, j, k]] = y[j];
                zz[[i, j, k]] = z[k];
            }
        }
    }

    (xx, yy, zz)
}

fn vdf_to_spherical_shells(
    vdf: &Array3<f64>,
    nshells: usize,
    ntheta: usize,
    nphi: usize,
) -> Array3<f64> {
    let r_values = get_linspace(0.0, 1.0, nshells);
    let theta_values = get_linspace(0.0, std::f64::consts::PI, ntheta);
    let phi_values = get_linspace(0.0, 2.0 * std::f64::consts::PI, nphi);

    //Remappings
    let (r, theta, phi) = meshgrid(&r_values, &theta_values, &phi_values);
    let x_sph = &r * theta.mapv(|x| x.sin()) * phi.mapv(|x| x.cos());
    let y_sph = &r * theta.mapv(|x| x.sin()) * phi.mapv(|x| x.sin());
    let z_sph = &r * theta.mapv(|x| x.cos());
    //Retvals
    let mut spherical_shells = Array3::<f64>::zeros((nshells, ntheta, nphi));

    for i in 0..nshells {
        for j in 0..ntheta {
            for k in 0..nphi {
                let x_point = x_sph[[i, j, k]];
                let y_point = y_sph[[i, j, k]];
                let z_point = z_sph[[i, j, k]];

                let x_index = ((x_point + 1.0) * 24.0).round() as usize; // Scale and shift x to match grid indices
                let y_index = ((y_point + 1.0) * 24.0).round() as usize; // Scale and shift y to match grid indices
                let z_index = ((z_point + 1.0) * 24.0).round() as usize; // Scale and shift z to match grid indices
                if x_index < VDF_SIZE && y_index < VDF_SIZE && z_index < VDF_SIZE {
                    spherical_shells[[i, j, k]] = vdf[[x_index, y_index, z_index]];
                }
            }
        }
    }
    spherical_shells
}

fn spherical_shells_to_vdf(
    spherical_shells: &Array3<f64>,
    nshells: usize,
    ntheta: usize,
    nphi: usize,
) -> Array3<f64> {
    let r_values = get_linspace(0.0, 1.0, nshells);
    let theta_values = get_linspace(0.0, std::f64::consts::PI, ntheta);
    let phi_values = get_linspace(0.0, 2.0 * std::f64::consts::PI, nphi);
    let (r, theta, phi) = meshgrid(&r_values, &theta_values, &phi_values);
    //Remappingfs
    let x_sph = &r * theta.mapv(|x| x.sin()) * phi.mapv(|x| x.cos());
    let y_sph = &r * theta.mapv(|x| x.sin()) * phi.mapv(|x| x.sin());
    let z_sph = &r * theta.mapv(|x| x.cos());

    let x_cart = x_sph.mapv(|x| (x + 1.0) * 24.0);
    let y_cart = y_sph.mapv(|y| (y + 1.0) * 24.0);
    let z_cart = z_sph.mapv(|z| (z + 1.0) * 24.0);

    let x_cart_r = x_cart.mapv(|x| x.round() as usize);
    let y_cart_r = y_cart.mapv(|y| y.round() as usize);
    let z_cart_r = z_cart.mapv(|z| z.round() as usize);
    //~Remappings

    //Retvals go here
    let mut vdf = Array3::<f64>::zeros((VDF_SIZE, VDF_SIZE, VDF_SIZE));

    for i in 0..nshells {
        for j in 0..ntheta {
            for k in 0..nphi {
                let x_index = x_cart_r[[i, j, k]];
                let y_index = y_cart_r[[i, j, k]];
                let z_index = z_cart_r[[i, j, k]];

                if x_index < 50 && y_index < 50 && z_index < 50 {
                    vdf[[x_index, y_index, z_index]] = spherical_shells[[i, j, k]];
                }
            }
        }
    }
    vdf
}

fn decompose_surface(mesh: &Array2<f64>, degree: usize) -> Vec<f32> {
    let sh = HarmonicsSet::new(degree, RealSH::Spherical);
    let mut coeffs: Vec<f32> = vec![0.0; sh.num_sh()];
    let shape = mesh.shape();
    for t in 0..shape[0] {
        for p in 0..shape[1] {
            let value = mesh[[t, p]];
            let theta = 180.0 - t as f32 * DTHETA;
            let phi = p as f32 * DPHI - 180.0;
            let coords = Coordinates::spherical(1.0, theta * DEG2RAD, phi * DEG2RAD);
            let local_coefficients = sh.eval(&coords);
            assert!(local_coefficients.len() == coeffs.len());
            for (i, &c) in local_coefficients.iter().enumerate() {
                coeffs[i] += c * value as f32 * (theta * DEG2RAD).sin();
            }
        }
    }

    /*
      Here, we would normally say that we sample in an unbiased way which would eliminate the sin(theta)
      factor in the intergal above. However we do not so the correct apporach is to include the sin(theta)
      factor and then to not use the factor 1/4*PI below. For more info read pages 15-17 of the citation.
      https://basesandframes.wordpress.com/wp-content/uploads/2016/05/spherical_harmonic_lighting_gritty_details_green_2003.pdf
    */

    let weight = 1.0; //4.0 * std::f32::consts::PI / 180.0;
    let nsamples = shape[0] * shape[1];
    let factor = weight / (nsamples as f32);
    coeffs.iter_mut().for_each(|val| *val *= factor);
    coeffs
}

fn decompose_spherical_slice(mut vdf_slice: Array2<f64>, degree: usize) -> (f64, Vec<f32>) {
    let vdf_clone = vdf_slice.clone();
    let max = vdf_clone.max().unwrap();
    vdf_slice.map_inplace(|x| *x = *x / max);
    let coeffs = decompose_surface(&vdf_slice, degree);
    let factor = *max as f64;
    (factor, coeffs)
}

fn decompose(spherical_shells: &Array3<f64>, degree: usize) -> (Vec<f64>, Vec<Vec<f32>>) {
    let shape = spherical_shells.shape();
    let mut coeffs: Vec<Vec<f32>> = Vec::with_capacity(shape[0]);
    let mut factors: Vec<f64> = Vec::with_capacity(shape[0]);
    for r in 0..shape[0] {
        let slice = spherical_shells.index_axis(Axis(0), r).to_owned();
        let (factor, coeff) = decompose_spherical_slice(slice, degree);
        coeffs.push(coeff);
        factors.push(factor);
    }
    (factors, coeffs)
}

fn reconstruct_spherical_slice(
    vdf_slice: &mut ArrayViewMut2<f64>,
    factor: f64,
    coeffs: &Vec<f32>,
    degree: usize,
    ntheta: usize,
    nphi: usize,
) {
    let sh = HarmonicsSet::new(degree, RealSH::Spherical); // Initialize spherical harmonics set
    for t in 0..ntheta {
        for p in 0..nphi {
            let theta = 180.0 - t as f32 * DTHETA;
            let phi = p as f32 * DPHI - 180.0;
            let coords = Coordinates::spherical(1.0, theta * DEG2RAD, phi * DEG2RAD);
            let coefficients_at_point: Vec<f32> = sh.eval(&coords);
            let mut value = 0.0;
            assert!(coefficients_at_point.len() == coeffs.len());
            for (i, &c) in coeffs.iter().enumerate() {
                value += coefficients_at_point[i] as f32 * c;
            }
            vdf_slice[[t, p]] = factor * value as f64;
        }
    }
    let binding = vdf_slice.to_owned();
    let max = binding.max().unwrap();
    vdf_slice.map_inplace(|x| *x = *x * factor / max);
}

fn reconstruct(
    factors: &Vec<f64>,
    coeffs: &Vec<Vec<f32>>,
    degree: usize,
    nshells: usize,
    ntheta: usize,
    nphi: usize,
) -> Array3<f64> {
    let mut spherical_shells: Array3<f64> = Array3::<f64>::zeros((nshells, ntheta, nphi));
    let shape = spherical_shells.shape();
    assert!(nshells == shape[0]);
    for r in 0..nshells {
        reconstruct_spherical_slice(
            &mut spherical_shells.index_axis_mut(Axis(0), r),
            factors[r],
            &coeffs[r],
            degree,
            ntheta,
            nphi,
        );
    }
    spherical_shells
}

fn calculate_compression_ratio(original_vdf: &Array3<f64>, coeffs: &Vec<Vec<f32>>) -> f32 {
    let mut coeff_sz: usize = 0;
    for c in coeffs.iter() {
        coeff_sz += c.len() * std::mem::size_of::<f32>();
    }
    let shape = original_vdf.shape();
    let vdf_size = shape[0] * shape[1] * shape[2] * std::mem::size_of::<f64>();
    let ratio = vdf_size as f32 / coeff_sz as f32;
    ratio
}

fn vdf_nd_array_to_vec(vdf: &Array3<f64>) -> Vec<f64> {
    let mut vdf_buffer: Vec<f64> = Vec::with_capacity(VDF_SIZE * VDF_SIZE * VDF_SIZE);
    for k in 0..VDF_SIZE {
        for j in 0..VDF_SIZE {
            for i in 0..VDF_SIZE {
                vdf_buffer.push(vdf[[i, j, k]]);
            }
        }
    }
    vdf_buffer
}
//~ ******************* SPH COMPRESSION *************************************//

#[pyfunction]
fn compress_mlp(
    vdf_file: &str,
    fourier_order: usize,
    epochs: usize,
    n_layers: usize,
    n_neurons: usize,
) -> PyResult<Vec<f64>> {
    let mut vdf = match read_vdf_from_file(&vdf_file) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{:?}", err);
            panic!();
        }
    };
    scale_vdf(&mut vdf);
    let mut reconstructed = compress_vdf(&vdf, fourier_order, epochs, n_layers, n_neurons);
    unscale_vdf(&mut reconstructed);
    Ok(reconstructed)
}

#[pyfunction]
fn compress_sph(vdf_file: &str, degree: usize) -> PyResult<Vec<f64>> {
    let vdf = match read_vdf_from_file_to_nd(&vdf_file) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{:?}", err);
            panic!();
        }
    };

    let spherical_shells = vdf_to_spherical_shells(&vdf, NSHELLS, NTHETA, NPHI);
    let (factors, coeffs) = decompose(&spherical_shells, degree);
    let recosntructed = reconstruct(&factors, &coeffs, degree, NSHELLS, NTHETA, NPHI);
    // let reconstructed_vdf = spherical_shells_to_vdf(&spherical_shells, NSHELLS, NTHETA, NPHI);
    let reconstructed_vdf = spherical_shells_to_vdf(&recosntructed, NSHELLS, NTHETA, NPHI);
    let vdf_retval = vdf_nd_array_to_vec(&reconstructed_vdf);
    let ratio = calculate_compression_ratio(&vdf, &coeffs);
    println!("Compression ratio = {}x .", ratio);
    Ok(vdf_retval)
}

#[pymodule]
fn mlp_compress(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(compress_sph, m)?)?;
    Ok(())
}
