import os
import time

import numpy as np

import tensorflow as tf

from MAI.Model.ReCa.CapsuleNetwork import CapsNet
from MAI.Utils.Functions.MarginLoss import margin_loss
import MAI.Utils.ConflictingBundles as cb
from MAI.Utils.Params import LEARNING_RATES, IMG_SIZE

# Configurations for cluster
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# for r in range(len(physical_devices)):
#    tf.config.experimental.set_memory_growth(physical_devices[r], True)


def compute_loss(logits, y, reconstruction, x):
    """ The loss is the sum of the margin loss and the reconstruction loss
    """
    num_classes = tf.shape(logits)[1]

    loss = margin_loss(logits, tf.one_hot(y, num_classes))
    loss = tf.reduce_mean(loss)

    # Calculate reconstruction loss
    reconstruction_loss = 0

    loss = loss + reconstruction_loss

    return loss, reconstruction_loss


def compute_accuracy(logits, labels):
    predictions = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=1), tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def train(train_ds, test_ds, class_names):
    """ Train capsule networks mirrored on multiple gpu's
    """

    # Run training for multiple epochs mirrored on multiple gpus
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync


    # Create a checkpoint directory to store the checkpoints.
    ckpt_dir = os.path.join("Output", "ckpt/", "ckpt")

    train_writer = tf.summary.create_file_writer("%s/log/train" % "Output/")
    test_writer = tf.summary.create_file_writer("%s/log/test" % "Output/")

    with strategy.scope():
        model = CapsNet()
        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATES)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        # Define metrics
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        # Function for a single training step
        def train_step(inputs):
            x, y = inputs
            with tf.GradientTape() as tape:
                logits, reconstruction, layers = model(x, y)
                model.summary()
                loss, _ = compute_loss(logits, y, reconstruction, x)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            acc = compute_accuracy(logits, y)

            return loss, acc, (x, reconstruction)

        # Function for a single test step
        def test_step(inputs):
            x, y = inputs
            logits, reconstruction, _ = model(x, y)
            loss, _ = compute_loss(logits, y, reconstruction, x)

            test_loss.update_state(loss)
            acc = compute_accuracy(logits, y)

            pred = tf.math.argmax(logits, axis=1)
            #cm = tf.math.confusion_matrix(y, pred, num_classes=10)
            return acc

        # Define functions for distributed training
        def distributed_train_step(dataset_inputs):
            return strategy.run(train_step, args=(dataset_inputs,))

        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs, ))

        distributed_train_step = tf.function(distributed_train_step)
        distributed_test_step = tf.function(distributed_test_step)

        # Loop for multiple epochs
        conflicts_int = None
        step = 0
        max_acc = 0.0
        for epoch in range(300):
            ########################################
            # Train
            ########################################
            for data in train_ds:
                start = time.time()
                distr_loss, distr_acc, distr_imgs = distributed_train_step(
                    data)
                train_loss = tf.reduce_mean(
                    distr_loss.values) if num_replicas > 1 else distr_loss
                acc = tf.reduce_mean(
                    distr_acc.values) if num_replicas > 1 else distr_acc
                use_reconstruction = True
                # Logging
                if step % 100 == 0:
                    time_per_step = (time.time()-start) * 1000 / 100
                    print("TRAIN | epoch %d (%d): acc=%.4f, loss=%.4f | Time per step[ms]: %.2f" %
                          (epoch, step, acc, train_loss.numpy(), time_per_step), flush=True)

                    # Create some recon tensorboard images (only GPU 0)
                    if use_reconstruction:
                        x = distr_imgs[0].values[0] if num_replicas > 1 else distr_imgs[0]
                        recon_x = distr_imgs[1].values[0] if num_replicas > 1 else distr_imgs[1]
                        recon_x = tf.reshape(
                            recon_x, [-1, tf.shape(x)[1], tf.shape(x)[2], IMG_SIZE])
                        x = tf.reshape(
                            x, [-1, tf.shape(x)[1], tf.shape(x)[2], IMG_SIZE])
                        img = tf.concat([x, recon_x], axis=1)
                        with train_writer.as_default():
                            tf.summary.image(
                                "X & Recon",
                                img,
                                step=step,
                                max_outputs=3,)

                    with train_writer.as_default():
                        # Write scalars
                        tf.summary.scalar("General/Accuracy", acc, step=step)
                        tf.summary.scalar(
                            "General/Loss", train_loss.numpy(), step=step)
                    start = time.time()
                    train_writer.flush()

                step += 1

            ####################
            # Checkpointing
            if epoch % 31 == 0:
                checkpoint.save(ckpt_dir)

                # Measure conflicts
                conflicts = cb.bundle_entropy(
                    model, train_ds,
                    8, LEARNING_RATES,
                    len(class_names), 32,
                    True)
                conflicts_int = cb.conflicts_integral(
                    conflicts_int, conflicts, 300)
                print("Num. bundles: %.0f; Bundle entropy: %.5f" %
                      (conflicts[-1][0], conflicts[-1][1]), flush=True)

                # Tensorboard shows entropy at step t and csv file the
                # normalized integral of the bundle entropy
                with train_writer.as_default():
                    tf.summary.scalar(
                        "bundle/Num", conflicts[-1][0], step=epoch)
                    tf.summary.scalar("bundle/Entropy",
                                      conflicts[-1][1], step=epoch)

                # Log test results (for replica 0 only for activation map and reconstruction)
                test_acc = np.mean(test_acc)
                max_acc = test_acc if test_acc > max_acc else max_acc
                print("TEST | epoch %d (%d): acc=%.4f, loss=%.4f" %
                      (epoch, step, test_acc, test_loss.result()), flush=True)

                with test_writer.as_default():
                    #tf.summary.image("Confusion Matrix", cm_image, step=step)
                    tf.summary.scalar("General/Accuracy", test_acc, step=step)
                    tf.summary.scalar(
                        "General/Loss", test_loss.result(), step=step)
                test_loss.reset_states()
                test_writer.flush()

        return max_acc
