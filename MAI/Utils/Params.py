from tensorflow import keras

IMG_SIZE = 512
IMG_SIZE_2 = 512
NUM_CLASSES = 14
BATCH_SIZE = 2
NUM_EPOCHS = 100

DIMENSIONS = "14,14,28"
LAYERS = "42,42,14"
ITERATIONS = 2
MAKE_SKIPS = False
NO_SKIPS = 2
ROUTING = "sda"

LEARNING_RATES = 0.001

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

trans_learning_rate = 0.0001
trans_weight_decay = 1e-5
trans_image_size = 256  # We'll resize input images to this size
trans_image_size_2 = 128  # We'll resize input images to this size
trans_patch_size = 16  # Size of the patches to be extract from the input images
trans_patch_size_2 = 16
trans_num_patches = (IMG_SIZE // trans_patch_size) ** 2
trans_num_patches_2 = (trans_image_size_2 // trans_patch_size_2) ** 2
trans_projection_dim = 16
trans_num_heads = 16
trans_transformer_units = [
    trans_projection_dim * 2,
    trans_projection_dim,
]  # Size of the transformer layers
trans_transformer_layers = 3
trans_mlp_head_units = [64, 32]