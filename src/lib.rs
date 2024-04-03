use pyo3::prelude::*;
mod network;
use crate::network::network::Network;
use nalgebra::DMatrix;
use rand::distributions::Uniform;
use rand::Rng;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

const VDF_SIZE: usize = 50;
const MIN_VAL: f64 = 1e-16;

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

#[pyfunction]
fn compress(
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

#[pymodule]
fn mlp_compress(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    Ok(())
}
