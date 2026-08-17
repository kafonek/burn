use burn_backend::distributed::DistributedOps;

use crate::{CubeBackend, CubeRuntime};

#[cfg(feature = "std")]
use crate::ops::numeric::{self, zeros_client};
#[cfg(feature = "std")]
use burn_backend::{
    DeviceId, TensorMetadata,
    cubecl::dtype_to_elem_type,
    distributed::{CollectiveTensor, ReduceOperation},
    tensor::{Device, FloatTensor},
};

impl<R: CubeRuntime> DistributedOps<Self> for CubeBackend<R> {
    #[cfg(feature = "std")]
    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        let device = tensor.device.clone();
        let stream = tensor.handle.stream;

        // The gradient sync server calls this from its own thread, and an unpinned client takes
        // its stream from the calling thread. Everything below would then be issued on the
        // server thread's stream and ordered against the producing stream only through the
        // runtime's shared-binding analysis, which compares against the point a buffer was
        // allocated rather than the point it was last written. A gradient whose buffer was
        // allocated before an earlier collective of the same round and written after it reads
        // as already synchronized, so the reduction runs over a buffer the producing kernel has
        // not finished writing. Pinning to the producing stream removes the cross-stream step:
        // the fence the collective records there sits behind every kernel already queued on it.
        let mut client = tensor.client.clone();
        // SAFETY: `handle.stream` is where the tensor was produced, on the same device as
        // `client`, and the pin is dropped with this local at the end of the call.
        unsafe { client.set_stream(stream) };

        let mut out_tensor = if tensor.handle.can_mut() && tensor.is_contiguous() {
            tensor
        } else {
            let zeros_tensor = zeros_client::<R>(
                client.clone(),
                device.clone(),
                tensor.shape(),
                tensor.dtype(),
            );
            numeric::add(zeros_tensor, tensor)
        };

        let op = match op {
            ReduceOperation::Sum => cubecl::server::ReduceOperation::Sum,
            ReduceOperation::Mean => cubecl::server::ReduceOperation::Mean,
        };

        client.all_reduce(
            out_tensor.handle.clone(),
            out_tensor.handle.clone(),
            dtype_to_elem_type(out_tensor.dtype),
            device_ids.clone(),
            op,
        );

        // The pin belongs to the collective, not to the tensor the caller gets back.
        out_tensor.client = R::client(&device);

        CollectiveTensor::new(out_tensor)
    }

    #[cfg(feature = "std")]
    fn sync_collective(device: &Device<Self>) {
        let client = R::client(device);
        client.sync_collective();
    }
}
