{
    "train_data_path": std.extVar('train_path'),
    "validation_data_path": std.extVar('val_path'),
    "dataset_reader": {
        "type": "norec",
    },
    "model": {
        "type": "hierarchical_doc",
        "embed_dim": 300,
        "word_encoder": {
            "type": "gru",
            "input_size": 300,
            "hidden_size": 200,
            "bidirectional": true
        },
        "sent_encoder": {
            "type": "gru",
            "input_size": 400,
            "hidden_size": 200,
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
        "batch_size": 32,
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 20,
    },
    "pytorch_seed": null,
    "numpy_seed": null,
    "random_seed": null,
}