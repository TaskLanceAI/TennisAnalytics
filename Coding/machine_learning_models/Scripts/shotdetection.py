
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import glob as glob
import numpy as np
import tensorflow as tf
from datetime import datetime

image_count = len(list(glob.glob('/content/gdrive/My Drive/tennis/*/*.png')))
print(image_count)

forehand = list(glob.glob('/content/gdrive/My Drive/tennis/forehand/*.png'))

img1 = PIL.Image.open(str(forehand[10]))
img1_resized = img1.resize((180, 180))
plt.imshow(img1_resized)

backhand = list(glob.glob('/content/gdrive/My Drive/tennis/backhand/*.png'))

img2 = PIL.Image.open(str(backhand[0]))
img2_resized = img2.resize((180, 180))
plt.imshow(img2_resized)

serve = list(glob.glob('/content/gdrive/My Drive/tennis/serve/*.png'))

img3 = PIL.Image.open(str(serve[10]))
img3_resized = img3.resize((180, 180))
plt.imshow(img3_resized)

"""Set the batch size, image height and width"""

batch_size = 32
img_height = 180
img_width = 180
data_dir = '/content/gdrive/My Drive/tennis'

"""Setup the training dataset"""

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

"""Setup validation dataset"""

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Sets up a timestamped log directory.
logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

print("Shape: ", image_batch[0].shape)

# Reshape the image for the Summary API.
img1 = np.reshape(image_batch[0], (1, 180, 180, 3))

# Using the file writer, log the reshaped image.
with file_writer.as_default():
  tf.summary.image("Training data", img1, step=0)

# Commented out IPython magic to ensure Python compatibility.
!kill 3396
# %tensorboard --logdir logs/train_data

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

"""The model consists of three convolution blocks with a max pool layer in each of them. There's a fully connected layer with 128 units on top of it that is activated by a relu activation function."""

num_classes = 3

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

features, labels = next(iter(train_ds))

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=0.0003)
# optimizer = keras.optimizers.SGD(learning_rate=1e-3)

# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

"""Set up summary writers to write the summaries to disk"""

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

import time

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

epochs = 20
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Iterate over the batches of the dataset.
    for step,  (x_batch_train, y_batch_train) in enumerate(train_ds):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y_batch_train, model(x_batch_train, training=True))

    with train_summary_writer.as_default():
      tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
      tf.summary.scalar('accuracy', epoch_accuracy.result(), step=epoch)


    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    epoch_loss_avg.reset_states()
    epoch_accuracy.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_ds:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/gradient_tape

model.summary()

"""Visualize training results"""

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

"""Predict on new data"""

test_path = "/content/gdrive/My Drive/test/test8.png"


img1 = PIL.Image.open(test_path)
img1_resized = img1.resize((180, 180))
plt.imshow(img1_resized)

img = keras.preprocessing.image.load_img(
    test_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

