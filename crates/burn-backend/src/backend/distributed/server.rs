use core::marker::PhantomData;
use std::collections::HashMap;

use crate::{Backend, TensorMetadata};
use crate::{DeviceId, DeviceOps, tensor::Device};

use crate::distributed::{
    DistributedConfig, DistributedParamId, DistributedParams, ReduceOperation, TensorRef,
    client::DistributedSyncMessage,
};

/// The backend work the gradient sync server performs, kept apart from the scheduling it does.
///
/// Every [Backend] supplies this through [BackendSyncOps]. The split lets the scheduling run
/// against a stand-in, so it can be tested without a backend and without a device.
pub(crate) trait SyncOps: 'static {
    /// A device taking part in the gradient sync.
    type Device: Clone + Send + 'static;

    /// A reference to a gradient that is waiting for its all_reduce.
    type Tensor: Clone + Send + 'static;

    /// Returns the [id](DeviceId) of the device.
    fn device_id(device: &Self::Device) -> DeviceId;

    /// Performs an in-place all_reduce of one parameter's gradients, over the devices that
    /// submitted them.
    fn all_reduce(tensors: &[Self::Tensor], op: ReduceOperation);

    /// Waits for the collective operations of the device to complete.
    fn sync_collective(device: &Self::Device);
}

/// The [SyncOps] of a [Backend].
pub(crate) struct BackendSyncOps<B: Backend> {
    _backend: PhantomData<B>,
}

impl<B: Backend> SyncOps for BackendSyncOps<B> {
    type Device = Device<B>;
    type Tensor = TensorRef<B>;

    fn device_id(device: &Self::Device) -> DeviceId {
        device.id()
    }

    fn all_reduce(tensors: &[Self::Tensor], op: ReduceOperation) {
        // Safety: Tensors sent to the `DistributedSyncServer` should not be accessed or modified until the end of the backward pass.
        let device_ids = tensors
            .iter()
            .map(|t| unsafe { &*t.0 }.device().id())
            .collect::<Vec<_>>();
        let reduced_tensors: Vec<B::FloatTensorPrimitive> = tensors
            .iter()
            .map(|tensor|
                // Safety: we can call `assume_resolved` on these tensors since we know `B::sync_collective` is called
                // at the end of the backward pass.
                unsafe {
                B::all_reduce((*tensor.0).clone(), op, device_ids.clone()).assume_resolved()
            })
            .collect();

        // Make the tensor reference point to the reduced tensor to perform an in-place all_reduce.
        // Safety: `B::sync_collective` should be automatically called after the backward pass.
        unsafe {
            tensors
                .iter()
                .zip(reduced_tensors)
                .for_each(|(tensor_ref, reduced_tensor)| *tensor_ref.0 = reduced_tensor);
        }
    }

    fn sync_collective(device: &Self::Device) {
        B::sync_collective(device)
    }
}

pub(crate) struct DistributedSyncServer<S: SyncOps> {
    config: DistributedConfig,
    all_reduce_ops_queue: HashMap<DistributedParamId, Vec<S::Tensor>>,
    param_required_map: HashMap<DistributedParamId, usize>,
    num_devices: usize,
    devices_registered: usize,
    syncing_devices: Vec<S::Device>,
    callbacks: HashMap<DeviceId, oneshot::Sender<Box<dyn FnOnce() + Send>>>,
}

impl<S: SyncOps> DistributedSyncServer<S> {
    /// Create a new gradient sync server instance.
    pub(crate) fn new(num_devices: usize, config: DistributedConfig) -> Self {
        Self {
            config,
            all_reduce_ops_queue: HashMap::default(),
            param_required_map: HashMap::default(),
            num_devices,
            devices_registered: 0,
            syncing_devices: vec![],
            callbacks: HashMap::default(),
        }
    }

    /// Process message from client.
    pub(crate) fn process_message(&mut self, msg: DistributedSyncMessage<S>) {
        match msg {
            DistributedSyncMessage::RegisterSyncParameters(params) => {
                self.register_sync_params(params)
            }
            DistributedSyncMessage::TensorSync((tensor, params)) => {
                self.register_tensor(tensor, params)
            }
            DistributedSyncMessage::CollectiveSync((device, callback)) => {
                self.collective_sync(device, callback)
            }
        }
    }

    /// Called at the start of the backward process. Lets the device announce what parameters are nodes in the autodiff graph and how many times they are required.
    fn register_sync_params(&mut self, sharded_params: Vec<DistributedParams>) {
        sharded_params.iter().for_each(|params| {
            *self.param_required_map.entry(params.param_id).or_insert(0) += 1;
        });
        self.devices_registered += 1;
    }

    /// Called on registration of a gradient. Calls the all_reduce operation for any parameter that is no longer required in the autodiff graph.
    fn register_tensor(&mut self, tensor: S::Tensor, sharded_params: DistributedParams) {
        let op_queue = self
            .all_reduce_ops_queue
            .entry(sharded_params.param_id)
            .or_insert(vec![]);
        op_queue.push(tensor.clone());
        self.launch_ops();
    }

    fn collective_sync(
        &mut self,
        device: S::Device,
        callback: oneshot::Sender<Box<dyn FnOnce() + Send>>,
    ) {
        self.callbacks.insert(S::device_id(&device), callback);
        self.syncing_devices.push(device);
        self.try_launch_sync();
    }

    fn try_launch_sync(&mut self) {
        // A device released before its peers arrive runs the next backward pass and registers
        // its parameters into the round that is still closing. `devices_registered` then
        // overshoots `num_devices`, `launch_ops` never matches again, and every device blocks
        // forever. So release nobody until every device has reached the sync point.
        if !self.all_reduce_ops_queue.is_empty() || self.syncing_devices.len() < self.num_devices {
            return;
        }

        self.devices_registered = 0;
        self.param_required_map.clear();

        for d in core::mem::take(&mut self.syncing_devices) {
            let callback = self
                .callbacks
                .remove(&S::device_id(&d))
                .expect("Syncing device should have a callback");
            let closure = Box::new(move || S::sync_collective(&d));
            callback.send(closure).expect("Can send callback");
        }

        self.callbacks.clear();
    }

    fn launch_ops(&mut self) {
        if self.devices_registered == self.num_devices {
            let op = self.config.all_reduce_op;

            for (param_id, num_tensors) in self.param_required_map.clone() {
                let queued_tensors = self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);

                if num_tensors == queued_tensors.len() {
                    S::all_reduce(queued_tensors, op);

                    self.all_reduce_ops_queue.remove(&param_id).unwrap();
                    self.param_required_map.remove(&param_id).unwrap();
                    self.try_launch_sync();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    const DEVICES: usize = 4;
    const PARAMS: usize = 3;
    const ROUNDS: usize = 3;

    /// How much later each device reaches the sync point than the one before it.
    const SKEW: Duration = Duration::from_millis(2);

    /// A round takes microseconds, so any wait this long means the devices no longer progress.
    const DEADLOCK_TIMEOUT: Duration = Duration::from_secs(30);

    #[derive(Clone, Debug, Default, PartialEq)]
    struct TestDevice(u16);

    /// Stands in for the gradient handle that the server all_reduces. Every gradient of a round
    /// shares one log, so the test can check which devices each all_reduce covered.
    #[derive(Clone)]
    struct TestGradient {
        device: usize,
        all_reduces: Arc<Mutex<Vec<Vec<usize>>>>,
    }

    struct TestSyncOps;

    impl SyncOps for TestSyncOps {
        type Device = TestDevice;
        type Tensor = TestGradient;

        fn device_id(device: &Self::Device) -> DeviceId {
            DeviceId {
                type_id: 0,
                index_id: device.0,
            }
        }

        fn all_reduce(tensors: &[Self::Tensor], _op: ReduceOperation) {
            let devices = tensors.iter().map(|t| t.device).collect::<Vec<_>>();
            tensors[0].all_reduces.lock().unwrap().push(devices);
        }

        fn sync_collective(_device: &Self::Device) {}
    }

    /// The sync point must hold every device until all of them have reached it.
    ///
    /// A device released early starts its next backward pass and registers its parameters into
    /// the round that is still closing, which pushes `devices_registered` past `num_devices`.
    /// `launch_ops` compares those two for equality, so it never runs again and every device
    /// waits forever. The skew below makes device 0 reach the sync point first on every round,
    /// which is the interleaving that triggers the defect.
    #[test]
    fn sync_point_holds_every_device_until_all_arrive() {
        let config = DistributedConfig {
            all_reduce_op: ReduceOperation::Sum,
        };
        let mut server = DistributedSyncServer::<TestSyncOps>::new(DEVICES, config);

        let (tx, rx) = mpsc::channel::<DistributedSyncMessage<TestSyncOps>>();
        thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                server.process_message(msg);
            }
        });

        let param_ids = (0..PARAMS)
            .map(|_| DistributedParamId::new())
            .collect::<Vec<_>>();
        let all_reduces = Arc::new(Mutex::new(Vec::new()));
        let (done_tx, done_rx) = mpsc::channel();

        for device in 0..DEVICES {
            let tx = tx.clone();
            let done_tx = done_tx.clone();
            let param_ids = param_ids.clone();
            let all_reduces = all_reduces.clone();

            thread::spawn(move || {
                for _ in 0..ROUNDS {
                    let params = param_ids
                        .iter()
                        .map(|param_id| DistributedParams {
                            param_id: *param_id,
                        })
                        .collect::<Vec<_>>();
                    tx.send(DistributedSyncMessage::RegisterSyncParameters(params))
                        .unwrap();

                    for param_id in &param_ids {
                        let gradient = TestGradient {
                            device,
                            all_reduces: all_reduces.clone(),
                        };
                        tx.send(DistributedSyncMessage::TensorSync((
                            gradient,
                            DistributedParams {
                                param_id: *param_id,
                            },
                        )))
                        .unwrap();
                    }

                    thread::sleep(SKEW * device as u32);

                    let (callback_tx, callback_rx) = oneshot::channel();
                    tx.send(DistributedSyncMessage::CollectiveSync((
                        TestDevice(device as u16),
                        callback_tx,
                    )))
                    .unwrap();
                    callback_rx.recv().expect("Can receive callback")();
                }

                done_tx.send(device).unwrap();
            });
        }
        drop(done_tx);

        for _ in 0..DEVICES {
            done_rx
                .recv_timeout(DEADLOCK_TIMEOUT)
                .expect("Every device should complete every round");
        }

        let all_reduces = all_reduces.lock().unwrap();
        assert_eq!(all_reduces.len(), PARAMS * ROUNDS);

        for devices in all_reduces.iter() {
            let mut devices = devices.clone();
            devices.sort();
            assert_eq!(devices, (0..DEVICES).collect::<Vec<_>>());
        }
    }
}
