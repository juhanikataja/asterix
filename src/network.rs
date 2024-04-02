/* File:    network.rs
 * Author: Kostis Papadakis
 * Description: A Tiny~Deep Neural Network
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 * USA.
 * */
#![allow(dead_code)]
pub mod network {

    use std::fs::File;
    use std::io::Write;

    use nalgebra::DMatrix;
    use rand::prelude::SliceRandom;
    use rand::{distributions::Uniform, Rng};

    const ADAM_BETA1: f32 = 0.9;
    const ADAM_BETA2: f32 = 0.999;
    const ADAM_EPS: f32 = 1e-8;
    const ADAM_DECAY: f32 = 0.01;

    pub trait NeuralNetworkTrait<T>:
        nalgebra::RealField
        + num_traits::Num
        + From<f32>
        + rand::distributions::uniform::SampleUniform
        + std::marker::Copy
    {
    }

    impl<T, U> NeuralNetworkTrait<U> for T where
        T: nalgebra::RealField
            + num_traits::Num
            + From<f32>
            + rand::distributions::uniform::SampleUniform
            + std::marker::Copy
    {
    }

    #[derive(Debug)]
    struct Layer<T> {
        /*
            w->  weights
            b->  bias
            z->  result of Ax+b
            a->  result of act(z)
            delta -> used in backprop
            dw-> weights' gradients
            db-> biases' gradients
            opt_v-> weight velocity
            opt_m-> weight momentum
        */
        w: DMatrix<T>,
        b: DMatrix<T>,
        z: DMatrix<T>,
        a: DMatrix<T>,
        dw: DMatrix<T>,
        db: DMatrix<T>,
        delta: DMatrix<T>,
        v: DMatrix<T>,
        m: DMatrix<T>,
        neurons: usize,
    }

    impl<T> Layer<T>
    where
        T: NeuralNetworkTrait<T>,
    {
        pub fn new(neurons: usize, input_size: usize) -> Self {
            let mut rng = rand::thread_rng();
            let range1 = Uniform::<T>::new(-Into::<T>::into(0.5), Into::<T>::into(0.5));
            let range2 = Uniform::<T>::new(-Into::<T>::into(0.5), Into::<T>::into(0.5));
            Layer {
                w: DMatrix::from_fn(input_size, neurons, |_, _| rng.sample(&range1)),
                b: DMatrix::from_fn(1, neurons, |_, _| rng.sample(&range2)),
                z: DMatrix::<T>::zeros(1, neurons),
                a: DMatrix::<T>::zeros(1, neurons),
                dw: DMatrix::<T>::zeros(input_size, neurons),
                db: DMatrix::<T>::zeros(1, neurons),
                delta: DMatrix::<T>::zeros(1, neurons),
                v: DMatrix::<T>::zeros(input_size, neurons),
                m: DMatrix::<T>::zeros(input_size, neurons),
                neurons: neurons,
            }
        }

        fn total_bytes(&self) -> usize {
            let mut total_size: usize = 0;
            total_size += self.w.ncols() * self.w.nrows();
            total_size += self.b.ncols() * self.b.nrows();
            total_size * std::mem::size_of::<T>()
        }

        pub fn reset_deltas(&mut self) {
            self.dw.fill(T::zero());
            self.db.fill(T::zero());
            self.delta.fill(T::zero());
        }

        fn tanh_mut(val: &mut T) {
            let ex: T = T::exp(*val);
            let exn: T = T::exp(-*val);
            *val = (ex - exn) / (ex + exn);
        }

        fn tanh(val: T) -> T {
            let ex: T = T::exp(val);
            let exn: T = T::exp(-val);
            (ex - exn) / (ex + exn)
        }

        fn tanh_prime_mut(val: &mut T) {
            *val = T::one() - *val * *val;
        }

        fn tanh_prime(val: T) -> T {
            T::one() - val * val
        }

        fn leaky_relu_mut(val: &mut T) {
            *val = if *val > T::zero() { *val } else { 0.001.into() };
        }

        fn leaky_relu(val: T) -> T {
            if val > T::zero() {
                val
            } else {
                0.001.into()
            }
        }

        fn leaky_relu_prime_mut(val: &mut T) {
            if *val > T::zero() {
                *val = T::one();
            } else {
                *val = 0.001.into();
            }
        }

        fn leaky_relu_prime(val: T) -> T {
            if val > T::zero() {
                T::one()
            } else {
                0.001.into()
            }
        }

        pub fn activate(val: T) -> T {
            Layer::leaky_relu(val)
        }
        pub fn activate_mut(val: &mut T) {
            Layer::leaky_relu_mut(val);
        }

        pub fn activate_prime(val: T) -> T {
            Layer::leaky_relu_prime(val)
        }

        pub fn activate_prime_mut(val: &mut T) {
            Layer::leaky_relu_prime_mut(val);
        }

        pub fn activate_prime_matrix(matrix: &DMatrix<T>) -> DMatrix<T> {
            matrix.map(|x| Self::activate_prime(x))
        }

        pub fn forward_matrix(&mut self, input: &DMatrix<T>) {
            input.mul_to(&self.w, &mut self.z);
            self.z += &self.b;
            self.a = self.z.clone();
            self.a.apply(|x| Self::activate_mut(x));
        }
    }

    #[derive(Debug)]
    pub struct Network<T> {
        layers: Vec<Layer<T>>,
        input_data: DMatrix<T>,
        output_data: DMatrix<T>,
        // input_buffer :mut DMatrix::<T>,
        // output_buffer :mut DMatrix::<T>
    }

    impl<T> Network<T>
    where
        T: NeuralNetworkTrait<T>,
    {
        pub fn new(
            input_size: usize,
            output_size: usize,
            num_layers: usize,
            neurons_per_layer: usize,
            input: &DMatrix<T>,
            output: &DMatrix<T>,
        ) -> Self {
            let mut v: Vec<Layer<T>> = vec![];
            v.push(Layer::new(neurons_per_layer, input_size));
            for _ in 1..num_layers - 1 {
                v.push(Layer::new(neurons_per_layer, neurons_per_layer));
            }
            v.push(Layer::new(output_size, neurons_per_layer));
            assert_eq!(input_size, input.ncols());
            assert_eq!(output_size, output.ncols());
            Network {
                layers: v,
                input_data: input.clone(),
                output_data: output.clone(),
            }
        }

        fn forward_matrix(&mut self, input: &DMatrix<T>) {
            //Layer 0 forwards the actual input
            self.layers[0].forward_matrix(&input);
            //Following layers forard their previouse's layer .a matrix
            for i in 1..self.layers.len() {
                let inp = self.layers[i - 1].a.clone();
                self.layers[i].forward_matrix(&inp);
            }
        }

        fn reset_deltas(&mut self) {
            for layer in self.layers.iter_mut() {
                layer.reset_deltas();
            }
        }

        pub fn train(&mut self, lr: T, epoch: usize) -> T {
            let ncols = self.input_data.ncols();
            let num_samples = self.input_data.nrows() as f32;
            let ncols_out = self.output_data.ncols();
            let mut buffer = DMatrix::<T>::zeros(1, ncols);
            let mut target = DMatrix::<T>::zeros(1, ncols_out);
            let mut running_cost: T = T::zero();
            self.shuffle_network_data();
            for i in 0..self.input_data.nrows() {
                self.reset_deltas();
                for j in 0..ncols {
                    buffer[(0, j)] = self.input_data[(i, j)];
                }
                for j in 0..ncols_out {
                    target[(0, j)] = self.output_data[(i, j)];
                }
                self.forward_matrix(&buffer);
                let output_error = self.get_output_gradient(&target);
                running_cost += output_error.sum().powi(2) as T;
                self.backward(&buffer, &output_error, T::one());
                // self.optimize(lr);
                self.optimize_adamw(epoch + 1, lr);
            }
            running_cost / num_samples.into()
        }

        pub fn train_batch(&mut self, lr: T, epoch: usize, batch: usize) -> T {
            let ncols = self.input_data.ncols();
            let num_samples = self.input_data.nrows() as f32;
            let ncols_out = self.output_data.ncols();
            let mut buffer = DMatrix::<T>::zeros(1, ncols);
            let mut target = DMatrix::<T>::zeros(1, ncols_out);
            let mut running_cost: T = T::zero();
            self.shuffle_network_data();
            for i in (0..self.input_data.nrows()).step_by(batch) {
                self.reset_deltas();
                for ii in i..i + batch {
                    if ii >= self.input_data.nrows() {
                        break;
                    }
                    for j in 0..ncols {
                        buffer[(0, j)] = self.input_data[(ii, j)];
                    }
                    for j in 0..ncols_out {
                        target[(0, j)] = self.output_data[(ii, j)];
                    }
                    self.forward_matrix(&buffer);
                    let output_error = self.get_output_gradient(&target);
                    running_cost += output_error.sum().powi(2) as T;
                    self.backward(&buffer, &output_error, T::one());
                    // self.optimize(lr);
                }
                self.scale(batch);
                self.optimize_adamw(epoch + 1, lr);
            }
            running_cost / num_samples.into()
        }

        fn zero_optimizer(&mut self) {
            for l in self.layers.iter_mut() {
                l.m.fill(T::zero());
                l.v.fill(T::zero());
            }
        }

        fn scale(&mut self, n: usize) {
            let nn = n as f32;
            for l in self.layers.iter_mut() {
                l.dw /= nn.into();
                l.db /= nn.into();
            }
        }

        fn optimize(&mut self, lr: T) {
            for l in self.layers.iter_mut() {
                l.w -= l.dw.scale(lr);
                l.b -= l.db.scale(lr);
            }
        }

        fn optimize_adam(&mut self, iteration: usize, lr: T) {
            for l in self.layers.iter_mut() {
                l.m = l.m.scale(ADAM_BETA1.into()) + l.dw.scale(T::one() - ADAM_BETA1.into());
                l.v = l.v.scale(ADAM_BETA2.into())
                    + l.dw
                        .map(|x| T::powi(x, 2))
                        .scale(T::one() - ADAM_BETA2.into());
                let m_hat = &l.m / (T::one() - T::powi(ADAM_BETA1.into(), iteration as i32));
                let mut v_hat = &l.v / (T::one() - T::powi(ADAM_BETA2.into(), iteration as i32));
                v_hat.apply(|x| *x = T::sqrt(*x));
                v_hat.add_scalar_mut(ADAM_EPS.into());
                let update = ((&m_hat).component_div(&v_hat)).scale(lr);
                l.w -= update;
                l.b -= l.db.scale(lr);
            }
        }

        fn optimize_adamw(&mut self, iteration: usize, lr: T) {
            for l in self.layers.iter_mut() {
                l.m = l.m.scale(ADAM_BETA1.into()) + l.dw.scale(T::one() - ADAM_BETA1.into());
                l.v = l.v.scale(ADAM_BETA2.into())
                    + l.dw
                        .map(|x| T::powi(x, 2))
                        .scale(T::one() - ADAM_BETA2.into());
                let m_hat = &l.m / (T::one() - T::powi(ADAM_BETA1.into(), iteration as i32));
                let mut v_hat = &l.v / (T::one() - T::powi(ADAM_BETA2.into(), iteration as i32));
                v_hat.apply(|x| *x = T::sqrt(*x));
                v_hat.add_scalar_mut(ADAM_EPS.into());
                let decay = &l.w.scale(ADAM_DECAY.into());
                let update = &m_hat.component_div(&v_hat) + decay;
                l.w = &l.w - &update.scale(lr);
                l.b = &l.b - &l.db.scale(lr);
            }
        }

        fn backward(&mut self, input: &DMatrix<T>, output_error: &DMatrix<T>, scale: T) {
            let mut delta_next = output_error.clone();

            for i in (1..self.layers.len()).rev() {
                let prev_activation = &self.layers[i - 1].a.clone();
                let activation_prime = Layer::activate_prime_matrix(&self.layers[i].z);

                self.layers[i].delta = delta_next.component_mul(&activation_prime);
                let delta = &self.layers[i].delta.clone();

                self.layers[i].dw += prev_activation.transpose() * delta;
                self.layers[i].db += delta;

                delta_next = &self.layers[i].delta * &self.layers[i].w.transpose();
            }

            // Handle the first layer separately
            let first_layer = &mut self.layers[0];
            let activation_prime = Layer::activate_prime_matrix(&first_layer.z);
            first_layer.delta = delta_next.component_mul(&activation_prime);
            first_layer.dw += input.transpose() * &first_layer.delta;
            first_layer.db += first_layer.delta.clone();

            for l in self.layers.iter_mut() {
                l.dw /= scale;
                l.db /= scale;
            }
        }

        pub fn eval(&mut self, sample: &DMatrix<T>, output_buffer: &mut DMatrix<T>) {
            self.forward_matrix(&sample);
            assert_eq!(output_buffer.ncols(), self.output_data.ncols());
            assert_eq!(sample.ncols(), self.input_data.ncols());
            let nn_output = &self.layers.last().unwrap().z;
            *output_buffer = nn_output.clone();
        }

        fn get_output_gradient(&mut self, target: &DMatrix<T>) -> DMatrix<T> {
            &self.layers.last().unwrap().a - target
        }

        pub fn calculate_total_bytes(&self) -> usize {
            let mut total_size: usize = 0;
            for l in self.layers.iter() {
                total_size += l.total_bytes();
            }
            total_size
        }

        pub fn get_state(&self) -> Vec<u8> {
            let total_bytes = self.calculate_total_bytes()
                + std::mem::size_of::<usize>() * (self.layers.len() + 1);
            let mut data: Vec<u8> = vec![];
            data.reserve_exact(total_bytes);
            let num_layers = self.layers.len();
            let bytes: [u8; std::mem::size_of::<usize>()] = num_layers.to_ne_bytes();
            data.extend_from_slice(&bytes);
            for l in self.layers.iter() {
                let n = l.neurons;
                let bytes: [u8; std::mem::size_of::<usize>()] = n.to_ne_bytes();
                data.extend_from_slice(&bytes);

                //Weights
                for j in 0..l.w.ncols() {
                    for i in 0..l.w.nrows() {
                        let bytes: [u8; std::mem::size_of::<f64>()] =
                            unsafe { std::mem::transmute(&l.w[(i, j)]) };
                        data.extend_from_slice(&bytes);
                    }
                }

                //Biases
                for j in 0..l.b.ncols() {
                    for i in 0..l.b.nrows() {
                        let bytes: [u8; std::mem::size_of::<f64>()] =
                            unsafe { std::mem::transmute(&l.b[(i, j)]) };
                        data.extend_from_slice(&bytes);
                    }
                }
            }
            println!("Bytes serialized {}/{}.", data.len(), total_bytes);
            assert!(
                data.len() == total_bytes,
                "Bytes mismatch during get_state()!"
            );
            data
        }

        pub fn save(&self, filename: &str) {
            println!("Storing state to {}", filename);
            let bytes = self.get_state();
            let mut file = File::create(filename).expect("Failed to create file");
            file.write_all(&bytes)
                .expect("Failed to write state to file!");
        }

        fn shuffle_network_data(&mut self) {
            let mut vec_in: Vec<Vec<T>> = self
                .input_data
                .row_iter()
                .map(|r| r.iter().copied().collect())
                .collect();

            let mut vec_out: Vec<Vec<T>> = self
                .output_data
                .row_iter()
                .map(|r| r.iter().copied().collect())
                .collect();

            let mut indices: Vec<usize> = (0..vec_in.len()).collect();
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);

            let mut shuffled_vec1: Vec<Vec<T>> = Vec::with_capacity(vec_in.len());
            let mut shuffled_vec2 = Vec::with_capacity(vec_in.len());

            for &i in &indices {
                shuffled_vec1.push(vec_in[i].clone());
                shuffled_vec2.push(vec_out[i].clone());
            }

            vec_in = shuffled_vec1;
            vec_out = shuffled_vec2;

            self.input_data =
                DMatrix::from_fn(vec_in.len() as _, vec_in[0].len() as _, |i, j| vec_in[i][j]);

            self.output_data =
                DMatrix::from_fn(vec_out.len() as _, vec_out[0].len() as _, |i, j| {
                    vec_out[i][j]
                });
        }
    }
}
