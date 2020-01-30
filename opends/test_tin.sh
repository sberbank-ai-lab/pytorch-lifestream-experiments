python metric_learning.py params.device="cuda:2" params.train.split_strategy.cnt_min=12 params.train.split_strategy.cnt_min=25 params.rnn.hidden_size=120 params.train.n_epoch=50 --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json

python ml_inference.py params.device="cuda:2" dataset.clip_transactions.min_len=1000  dataset.clip_transactions.max_len=1000 output.path="../data/tinkoff/embeddings_1000"  trx_features.path="../data/tinkoff/trx_features_1000"  --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json
python ml_inference.py params.device="cuda:2" dataset.clip_transactions.min_len=100   dataset.clip_transactions.max_len=200  output.path="../data/tinkoff/embeddings_0200"  trx_features.path="../data/tinkoff/trx_features_0200"  --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json
python ml_inference.py params.device="cuda:2" dataset.clip_transactions.min_len=75    dataset.clip_transactions.max_len=150  output.path="../data/tinkoff/embeddings_0150"  trx_features.path="../data/tinkoff/trx_features_0150"  --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json
python ml_inference.py params.device="cuda:2" dataset.clip_transactions.min_len=50    dataset.clip_transactions.max_len=100  output.path="../data/tinkoff/embeddings_0100"  trx_features.path="../data/tinkoff/trx_features_0100"  --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json
python ml_inference.py params.device="cuda:2" dataset.clip_transactions.min_len=25    dataset.clip_transactions.max_len=50   output.path="../data/tinkoff/embeddings_0050"  trx_features.path="../data/tinkoff/trx_features_0050"  --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json
python ml_inference.py params.device="cuda:2" dataset.clip_transactions.min_len=12    dataset.clip_transactions.max_len=25   output.path="../data/tinkoff/embeddings_0025"  trx_features.path="../data/tinkoff/trx_features_0025"  --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json
python ml_inference.py params.device="cuda:2" dataset.clip_transactions.min_len=5     dataset.clip_transactions.max_len=10   output.path="../data/tinkoff/embeddings_0010"  trx_features.path="../data/tinkoff/trx_features_0010"  --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json

python -m scenario_tin_cls compare_approaches --ml_embedding_file_names \
    "embeddings_1000.pickle"  "trx_features_1000.pickle" \
    "embeddings_0200.pickle"  "trx_features_0200.pickle" \
    "embeddings_0150.pickle"  "trx_features_0150.pickle" \
    "embeddings_0100.pickle"  "trx_features_0100.pickle" \
    "embeddings_0050.pickle"  "trx_features_0050.pickle" \
    "embeddings_0025.pickle"  "trx_features_0025.pickle" \
    "embeddings_0010.pickle"  "trx_features_0010.pickle"
