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
    "embeddings_0200.pickle"  "trx_features_0200.pickle"  \
    "embeddings_0150.pickle"  "trx_features_0150.pickle"  \
    "embeddings_0100.pickle"  "trx_features_0100.pickle"  \
    "embeddings_0050.pickle"  "trx_features_0050.pickle"   \
    "embeddings_0025.pickle"  "trx_features_0025.pickle"   \
    "embeddings_0010.pickle"  "trx_features_0010.pickle"


gender_cd: linear_baseline_trx                                          0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: linear_embeds: embeddings_0010.pickle                        0.5662  0.5626  0.5698 0.0026  [0.563 0.566 0.566 0.567 0.570]                       0.5672  0.5657  0.5686 0.0011  [0.566 0.567 0.567 0.568 0.568]
gender_cd: linear_embeds: embeddings_0025.pickle                        0.5969  0.5882  0.6056 0.0063  [0.587 0.597 0.598 0.599 0.604]                       0.6079  0.6057  0.6101 0.0016  [0.606 0.607 0.607 0.609 0.610]
gender_cd: linear_embeds: embeddings_0050.pickle                        0.6194  0.6136  0.6253 0.0042  [0.613 0.619 0.619 0.620 0.625]                       0.6317  0.6286  0.6347 0.0022  [0.629 0.629 0.633 0.633 0.634]
gender_cd: linear_embeds: embeddings_0100.pickle                        0.6327  0.6257  0.6396 0.0050  [0.628 0.630 0.631 0.633 0.641]                       0.6476  0.6442  0.6509 0.0024  [0.644 0.647 0.648 0.648 0.651]
gender_cd: linear_embeds: embeddings_0150.pickle                        0.6366  0.6285  0.6446 0.0058  [0.631 0.633 0.634 0.639 0.646]                       0.6515  0.6502  0.6527 0.0009  [0.650 0.651 0.651 0.652 0.653]
gender_cd: linear_embeds: embeddings_0200.pickle                        0.6377  0.6287  0.6467 0.0065  [0.632 0.632 0.637 0.640 0.648]                       0.6521  0.6491  0.6552 0.0022  [0.649 0.652 0.652 0.654 0.654]
gender_cd: linear_embeds: embeddings_1000.pickle                        0.6398  0.6301  0.6495 0.0070  [0.633 0.634 0.640 0.643 0.649]                       0.6559  0.6548  0.6571 0.0008  [0.655 0.655 0.656 0.657 0.657]
gender_cd: linear_embeds: trx_features_0010.pickle                      0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: linear_embeds: trx_features_0025.pickle                      0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: linear_embeds: trx_features_0050.pickle                      0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: linear_embeds: trx_features_0100.pickle                      0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: linear_embeds: trx_features_0150.pickle                      0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: linear_embeds: trx_features_0200.pickle                      0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: linear_embeds: trx_features_1000.pickle                      0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: linear_random                                                0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]                       0.5000  0.5000  0.5000 0.0000  [0.500 0.500 0.500 0.500 0.500]
gender_cd: xgb_baseline_trx                                             0.6661  0.6612  0.6710 0.0035  [0.664 0.664 0.665 0.666 0.672]                       0.7036  0.7002  0.7070 0.0024  [0.701 0.701 0.704 0.705 0.707]
gender_cd: xgb_embeds: embeddings_0010.pickle                           0.5634  0.5595  0.5673 0.0028  [0.559 0.563 0.564 0.564 0.567]                       0.5684  0.5624  0.5744 0.0043  [0.563 0.566 0.567 0.571 0.574]
gender_cd: xgb_embeds: embeddings_0025.pickle                           0.5971  0.5889  0.6053 0.0059  [0.588 0.596 0.598 0.599 0.604]                       0.6128  0.6066  0.6190 0.0045  [0.609 0.611 0.612 0.612 0.620]
gender_cd: xgb_embeds: embeddings_0050.pickle                           0.6216  0.6170  0.6263 0.0033  [0.617 0.620 0.622 0.625 0.625]                       0.6361  0.6330  0.6392 0.0022  [0.633 0.635 0.636 0.638 0.639]
gender_cd: xgb_embeds: embeddings_0100.pickle                           0.6340  0.6237  0.6442 0.0074  [0.625 0.630 0.633 0.637 0.645]                       0.6531  0.6501  0.6561 0.0021  [0.651 0.651 0.653 0.654 0.656]
gender_cd: xgb_embeds: embeddings_0150.pickle                           0.6374  0.6303  0.6444 0.0051  [0.631 0.634 0.638 0.641 0.643]                       0.6557  0.6525  0.6589 0.0023  [0.653 0.654 0.656 0.658 0.658]
gender_cd: xgb_embeds: embeddings_0200.pickle                           0.6390  0.6319  0.6461 0.0051  [0.633 0.635 0.639 0.642 0.646]                       0.6602  0.6555  0.6650 0.0034  [0.658 0.658 0.658 0.661 0.666]
gender_cd: xgb_embeds: embeddings_1000.pickle                           0.6420  0.6341  0.6500 0.0057  [0.635 0.638 0.643 0.645 0.649]                       0.6606  0.6585  0.6626 0.0015  [0.659 0.659 0.660 0.662 0.662]
gender_cd: xgb_embeds: trx_features_0010.pickle                         0.5652  0.5612  0.5692 0.0029  [0.561 0.563 0.566 0.568 0.568]                       0.5704  0.5671  0.5737 0.0024  [0.567 0.570 0.571 0.572 0.573]
gender_cd: xgb_embeds: trx_features_0025.pickle                         0.6038  0.5961  0.6115 0.0056  [0.597 0.601 0.603 0.606 0.612]                       0.6244  0.6192  0.6296 0.0038  [0.618 0.624 0.625 0.626 0.629]
gender_cd: xgb_embeds: trx_features_0050.pickle                         0.6341  0.6274  0.6408 0.0048  [0.628 0.632 0.633 0.635 0.641]                       0.6585  0.6549  0.6620 0.0026  [0.655 0.656 0.659 0.661 0.661]
gender_cd: xgb_embeds: trx_features_0100.pickle                         0.6518  0.6484  0.6553 0.0025  [0.649 0.649 0.653 0.653 0.654]                       0.6850  0.6828  0.6871 0.0016  [0.683 0.684 0.685 0.686 0.686]
gender_cd: xgb_embeds: trx_features_0150.pickle                         0.6572  0.6520  0.6624 0.0037  [0.653 0.655 0.658 0.659 0.662]                       0.6893  0.6856  0.6930 0.0027  [0.687 0.688 0.688 0.689 0.694]
gender_cd: xgb_embeds: trx_features_0200.pickle                         0.6627  0.6567  0.6686 0.0043  [0.659 0.660 0.662 0.663 0.670]                       0.6934  0.6902  0.6965 0.0023  [0.690 0.692 0.694 0.695 0.695]
gender_cd: xgb_embeds: trx_features_1000.pickle                         0.6661  0.6612  0.6710 0.0035  [0.664 0.664 0.665 0.666 0.672]                       0.7036  0.7002  0.7070 0.0024  [0.701 0.701 0.704 0.705 0.707]
gender_cd: xgb_random                                                   0.4998  0.4987  0.5009 0.0008  [0.499 0.499 0.500 0.500 0.501]                       0.4999  0.4981  0.5017 0.0013  [0.499 0.499 0.500 0.501 0.501]
