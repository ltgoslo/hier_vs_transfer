{
    "train_data_path": std.extVar('train_path'),
    "validation_data_path": std.extVar('val_path'),
    "dataset_reader": {
        "type": "norec-flat",
        "num_examples": std.extVar('num_examples')
    },
     "model": {
    "type": "mycnn",
    "embed_dim": 300,
        "word_encoder": {
            "type": "cnn",
            "embedding_dim": 300,
            "num_filters": 50,
            "ngram_filter_sizes": [2, 3, 4, 5],
            "conv_layer_activation": "relu",
            "output_dim": 50,
        },
  },
  "iterator": {
        "type": "basic",
        "batch_size": 32,
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    }
  }
}
