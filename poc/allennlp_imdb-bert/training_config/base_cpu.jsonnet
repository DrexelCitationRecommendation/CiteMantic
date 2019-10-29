{
  "dataset_reader": {
    "type": "imdb",
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "do_lowercase": true,
        "use_starting_offsets": true
      }
    },
    "tokenizer": {
        "type": "character",
    }
  },
  "train_data_path": "train",
  "test_data_path": "test",
  "evaluate_on_test": true,
  "model": {
    "type": "rnn_classifier",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"]
        },
      "token_embedders": {
        "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased",
          "top_layer_only": true,
          "requires_grad": false
        }
      }
    },
    "seq2vec_encoder": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 768,
      "hidden_size": 100,
      "num_layers": 1
    },
    "dropout": 0.2
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 64
  },

  "trainer": {
    "num_epochs": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
    }
  }
}
