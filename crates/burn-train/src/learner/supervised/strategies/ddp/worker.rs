use crate::ddp::epoch::{DdpTrainEpoch, DdpValidEpoch};
use crate::ddp::strategy::WorkerComponents;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent};
use crate::single::TrainingLoop;
use crate::{
    Learner, LearnerModel, LearningCheckpointer, SupervisedTrainingEventProcessor, TrainLoader,
    ValidLoader,
};
use burn_core::tensor::Device;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A worker runs the model, syncing gradients using collective operations.
/// Event processing and validation is optional too.
pub(crate) struct DdpWorker<M: LearnerModel> {
    device: Device,
    device_index: usize,
    learner: Learner<M>,
    event_processor: Arc<Mutex<SupervisedTrainingEventProcessor<M>>>,
    components: WorkerComponents,
    checkpointer: Option<LearningCheckpointer<M>>,
    dataloader_train: TrainLoader<M>,
    dataloader_valid: Option<ValidLoader<M>>,
    starting_epoch: usize,
    peer_count: usize,
}

impl<M: LearnerModel> DdpWorker<M> {
    /// Starts a worker that runs the model in a data distributed parallel
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        device: Device,
        device_index: usize,
        learner: Learner<M>,
        event_processor: Arc<Mutex<SupervisedTrainingEventProcessor<M>>>,
        components: WorkerComponents,
        checkpointer: Option<LearningCheckpointer<M>>,
        dataloader_train: TrainLoader<M>,
        dataloader_valid: Option<ValidLoader<M>>,
        starting_epoch: usize,
        peer_count: usize,
    ) -> JoinHandle<M> {
        let worker = Self {
            device,
            device_index,
            learner,
            event_processor,
            components,
            checkpointer,
            dataloader_train,
            dataloader_valid,
            starting_epoch,
            peer_count,
        };

        // Thread names are truncated to 15 bytes on Linux, so keep this short.
        std::thread::Builder::new()
            .name(std::format!("ddp-worker-{device_index}"))
            .spawn(|| worker.fit())
            .expect("Failed to spawn distributed data parallel worker thread")
    }

    /// Fits the model,
    pub fn fit(mut self) -> M {
        // The worker on the first device is the main one: it validates and processes events.
        let is_main = self.device_index == 0;
        let num_epochs = self.components.num_epochs;
        let interrupter = self.components.interrupter;

        // Changed the train epoch to keep the dataloaders
        let epoch_train = DdpTrainEpoch::<M>::new(
            self.dataloader_train.clone(),
            self.components.grad_accumulation,
        );
        let epoch_valid = self
            .dataloader_valid
            .map(|dataloader| DdpValidEpoch::<M>::new(dataloader));
        self.learner.fork(&self.device);
        self.learner.grad_sharded();

        for training_progress in TrainingLoop::new(self.starting_epoch, num_epochs) {
            let epoch = training_progress.items_processed;

            if is_main {
                self.event_processor
                    .lock()
                    .unwrap()
                    .process_train(LearnerEvent::StartSplit {
                        epoch_number: epoch,
                        total_items: self.components.train_total_items,
                    });
            }

            epoch_train.run(
                &mut self.learner,
                &training_progress,
                self.event_processor.clone(),
                &interrupter,
                self.peer_count,
            );

            if is_main {
                self.event_processor
                    .lock()
                    .unwrap()
                    .process_train(LearnerEvent::EndSplit(epoch));
            }

            // Workers using early stopping must all reach the epoch barrier below. Validation will
            // observe the interruption and return promptly on the main worker.
            if interrupter.should_stop() && self.components.early_stopping.is_none() {
                break;
            }

            // Validation
            if let Some(runner) = &epoch_valid {
                {
                    self.event_processor
                        .lock()
                        .unwrap()
                        .process_valid(LearnerEvent::StartSplit {
                            epoch_number: epoch,
                            total_items: self.components.valid_total_items,
                        });
                }
                let mut event_processor = self.event_processor.lock().unwrap();
                runner.run(
                    &self.learner.model(),
                    &training_progress,
                    &mut event_processor,
                    &interrupter,
                );
                event_processor.process_valid(LearnerEvent::EndSplit(epoch));
                event_processor.process_train(LearnerEvent::EndEpoch(epoch));
            }

            if self.components.early_stopping.is_some() {
                // Only the main worker runs validation, so every worker waits for its validation
                // and epoch end events to be queued before draining the event processor. This way
                // early stopping never reads a missing or stale metric value.
                self.components.epoch_barrier.wait();
            }

            if self.checkpointer.is_some() || self.components.early_stopping.is_some() {
                self.event_processor.lock().unwrap().flush();
            }

            if interrupter.should_stop() {
                break;
            }

            if let Some(checkpointer) = &mut self.checkpointer {
                checkpointer.checkpoint(&self.learner, epoch, &self.components.event_store);
            }

            if let Some(early_stopping) = &mut self.components.early_stopping
                && early_stopping.should_stop(epoch, &self.components.event_store)
            {
                break;
            }
        }

        self.learner.model()
    }
}
