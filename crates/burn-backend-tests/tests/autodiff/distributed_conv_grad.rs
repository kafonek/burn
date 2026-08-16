//! Regression test: the gradient a DDP round returns for a Conv2d weight is intermittently
//! not the gradient.
//!
//! Nothing in the loop below changes between iterations. The weights and the input are rebuilt
//! from the same constants every iteration, there is no optimizer, and every device holds
//! bit-identical tensors. The weight gradient is therefore a deterministic function with one
//! correct value, and a mean reduction over identical peers cannot change it. Every iteration
//! must return the value the first iteration returned.
//!
//! On two CUDA devices in one process it does not. Two distinct failures show up:
//!
//! * the gradient is replaced by an unrelated tensor, the same one every time, whose norm has
//!   nothing to do with the parameter it lands on;
//! * the gradient is exactly half the correct value, as though one peer's contribution was
//!   dropped while the mean still divided by the peer count.
//!
//! Both ranks always observe the identical wrong value, so the wrong value survives the
//! collective rather than being a local read of a buffer still in flight.
//!
//! Two details are needed to reproduce it. The round has to carry several distributed
//! parameters; a single parameter per round never fails. And autotune has to be warm on every
//! device before the first collective, otherwise tuning a kernel while a peer spins in the
//! collective deadlocks before the first comparison is ever made.

use super::*;
use burn_tensor::{
    Device, DeviceType, Shape, TensorData,
    activation::gelu,
    distributed::{DistributedConfig, DistributedContext, ReduceOperation},
    module::conv2d,
    ops::ConvOptions,
};
use burn_backend::distributed::DistributedParamId;
use serial_test::serial;
use std::sync::{Arc, Barrier};

const LAYERS: usize = 5;
const CHANNELS: usize = 64;
const IMAGE: usize = 32;
const BATCH_SIZE: usize = 64;
const ITERATIONS: usize = 300;

/// A gradient that differs from the first iteration by more than this is not the gradient.
const TOLERANCE: f64 = 1e-3;

#[test]
#[serial]
fn ddp_returns_the_same_conv_weight_gradient_every_iteration() {
    let devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    if devices.len() < 2 {
        return;
    }

    // Tuning a kernel while a peer spins in the collective deadlocks, so every device is warmed
    // on the shapes under test before the sync server starts.
    for device in &devices {
        weight_gradient_norms(device, &weights(device), &input(device));
    }

    let _context = DistributedContext::init(
        devices.clone(),
        DistributedConfig {
            all_reduce_op: ReduceOperation::Mean,
        },
    );

    let barrier = Arc::new(Barrier::new(devices.len()));
    let handles: Vec<_> = devices
        .into_iter()
        .map(|device| {
            let barrier = barrier.clone();
            std::thread::spawn(move || peer(device, barrier))
        })
        .collect();

    let failures: Vec<String> = handles
        .into_iter()
        .flat_map(|handle| handle.join().expect("peer thread panicked"))
        .collect();

    assert!(
        failures.is_empty(),
        "the reduced gradient changed between iterations of an unchanging computation:\n{}",
        failures.join("\n")
    );
}

/// Runs the checked loop on one device and returns a line per disagreeing iteration.
fn peer(device: Device, barrier: Arc<Barrier>) -> Vec<String> {
    let input = input(&device);
    let mut expected: Option<Vec<f64>> = None;
    let mut failures = Vec::new();

    for iteration in 0..ITERATIONS {
        let weights: Vec<TestTensor<4>> = weights(&device)
            .into_iter()
            .enumerate()
            .map(|(layer, weight)| {
                weight.set_distributed(DistributedParamId::from(layer as u64 + 1))
            })
            .collect();
        let norms = weight_gradient_norms(&device, &weights, &input);

        match &expected {
            None => expected = Some(norms),
            Some(expected) => {
                for (layer, (norm, want)) in norms.iter().zip(expected).enumerate() {
                    let relative = (norm - want).abs() / want;
                    if relative > TOLERANCE {
                        failures.push(format!(
                            "  iteration {iteration}, layer {layer}: got {norm:e}, \
                             first iteration returned {want:e} (relative {relative:e})"
                        ));
                    }
                }
            }
        }

        barrier.wait();
    }

    failures
}

/// One forward and one backward over the stack, returning each weight gradient's L2 norm.
fn weight_gradient_norms(
    _device: &Device,
    weights: &[TestTensor<4>],
    input: &TestTensor<4>,
) -> Vec<f64> {
    let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
    let mut x = input.clone();
    for weight in weights {
        x = gelu(conv2d(x, weight.clone(), None, options.clone()));
    }
    let gradients = x.powi_scalar(2).mean().backward();

    weights
        .iter()
        .map(|weight| {
            let gradient = weight
                .grad(&gradients)
                .expect("every weight requires a gradient");
            (gradient.powi_scalar(2).sum().into_scalar::<f32>() as f64).sqrt()
        })
        .collect()
}

fn weights(device: &Device) -> Vec<TestTensor<4>> {
    (0..LAYERS)
        .map(|layer| {
            let data = TensorData::new(
                fill(CHANNELS * CHANNELS * 9, weight_scale(), 7 + layer as u64),
                Shape::new([CHANNELS, CHANNELS, 3, 3]),
            );
            TestTensor::<4>::from_data(data, device).require_grad()
        })
        .collect()
}

fn input(device: &Device) -> TestTensor<4> {
    let data = TensorData::new(
        fill(BATCH_SIZE * CHANNELS * IMAGE * IMAGE, 1.0, 11),
        Shape::new([BATCH_SIZE, CHANNELS, IMAGE, IMAGE]),
    );
    TestTensor::<4>::from_data(data, device)
}

/// Keeps the activation scale flat across the stack, so the gradients under test are ordinary
/// magnitudes rather than near-denormal values.
fn weight_scale() -> f32 {
    (3.0f32 / (CHANNELS * 9) as f32).sqrt()
}

/// Deterministic values, so every peer holds bit-identical tensors without seeding a device.
fn fill(len: usize, scale: f32, offset: u64) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u64 * 2654435761 + offset) % 1000;
            (x as f32 / 500.0 - 1.0) * scale
        })
        .collect()
}
