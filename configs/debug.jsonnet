{
    "train_data_path": std.extVar('train_path'),
    "validation_data_path": std.extVar('val_path'),
    "dataset_reader": {
        "type": "norec-hier",
    },
    "model": {
        "type": "hierarchical_doc",
        "embed_dim": 200,
        "word_encoder": {
            "type": "gru",
            "input_size": 200,
            "hidden_size": 50,
            "bidirectional": true
        },
        "sent_encoder": {
            "type": "gru",
            "input_size": 100,
            "hidden_size": 50,
            "bidirectional": true
        },
        "word_attn": {
            "type": "dot_product",
            "normalize": true
        },
        "sent_attn": {
            "type": "dot_product",
            "normalize": true
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 64,
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 1,
        "patience": 2,
        "cuda_device": -1,
    },
    "pytorch_seed": null,
    "numpy_seed": null,
    "random_seed": null,
}
