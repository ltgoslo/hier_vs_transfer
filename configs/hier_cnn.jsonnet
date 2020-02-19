{
    "train_data_path": std.extVar('train_path'),
    "validation_data_path": std.extVar('val_path'),
    "dataset_reader": {
        "type": "norec-hier",
        "num_examples": std.extVar('num_examples')
    },
    "model": {
        "type": "hierarchical_cnn",
        "embed_dim": 200,
        "word_encoder": {
            "type": "cnn",
            "embedding_dim": 200,
            "num_filters": 50,
            "ngram_filter_sizes": [2, 3, 4, 5],
            "conv_layer_activation": "relu",
            "output_dim": 50,
        },
        "sent_encoder": {
            "type": "cnn",
            "embedding_dim": 50,
            "num_filters": 50,
            "ngram_filter_sizes": [2, 3],
            "conv_layer_activation": "relu",
            "output_dim": 50,
        },
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8,
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 20,
        "patience": 2,
        "cuda_device": -1,
    },
    "pytorch_seed": null,
    "numpy_seed": null,
    "random_seed": null,
}
