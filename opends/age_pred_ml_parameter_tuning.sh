python metric_learning.py model_path.model="models/age_pred_ml_model_v_base.p"      params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json
python metric_learning.py model_path.model="models/age_pred_ml_model_v_hs512.p"     params.rnn.hidden_size=512 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json
python metric_learning.py model_path.model="models/age_pred_ml_model_v_epoch250.p"  params.rnn.hidden_size=256 params.train.n_epoch=250 params.train.batch_size=64  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json
python metric_learning.py model_path.model="models/age_pred_ml_model_v_bs32.p"      params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=32  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json
python metric_learning.py model_path.model="models/age_pred_ml_model_v_bs128.p"     params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=128 params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json
python metric_learning.py model_path.model="models/age_pred_ml_model_v_bs256.p"     params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=256 params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=150 --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json
python metric_learning.py model_path.model="models/age_pred_ml_model_v_len150.p"    params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=75  params.train.split_strategy.cnt_max=200 --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json
python metric_learning.py model_path.model="models/age_pred_ml_model_v_len60.p"     params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=25  params.train.split_strategy.cnt_max=75  --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json
python metric_learning.py model_path.model="models/age_pred_ml_model_v_len250.p"    params.rnn.hidden_size=256 params.train.n_epoch=150 params.train.batch_size=64  params.train.split_strategy.cnt_min=200 params.train.split_strategy.cnt_max=300 --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json

python ml_inference.py model_path.models=["models/age_pred_ml_model_v_base.p"    ] output.path="../data/age-pred/embeddings_v_base"     --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json
python ml_inference.py model_path.models=["models/age_pred_ml_model_v_hs512.p"   ] output.path="../data/age-pred/embeddings_v_hs512"    --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json
python ml_inference.py model_path.models=["models/age_pred_ml_model_v_epoch250.p"] output.path="../data/age-pred/embeddings_v_epoch250" --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json
python ml_inference.py model_path.models=["models/age_pred_ml_model_v_bs32.p"    ] output.path="../data/age-pred/embeddings_v_bs32"     --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json
python ml_inference.py model_path.models=["models/age_pred_ml_model_v_bs128.p"   ] output.path="../data/age-pred/embeddings_v_bs128"    --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json
python ml_inference.py model_path.models=["models/age_pred_ml_model_v_bs256.p"   ] output.path="../data/age-pred/embeddings_v_bs256"    --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json
python ml_inference.py model_path.models=["models/age_pred_ml_model_v_len150.p"  ] output.path="../data/age-pred/embeddings_v_len150"   --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json
python ml_inference.py model_path.models=["models/age_pred_ml_model_v_len60.p"   ] output.path="../data/age-pred/embeddings_v_len60"    --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json
python ml_inference.py model_path.models=["models/age_pred_ml_model_v_len250.p"  ] output.path="../data/age-pred/embeddings_v_len250"   --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json

python -m scenario_age_pred compare_approaches --pos 4 5 6 7 8 9 10 11 12 --ml_embedding_file_names \
    "embeddings_v_base.pickle"     \
    "embeddings_v_hs512.pickle"    \
    "embeddings_v_epoch250.pickle" \
    "embeddings_v_bs32.pickle"     \
    "embeddings_v_bs128.pickle"    \
    "embeddings_v_bs256.pickle"    \
    "embeddings_v_len150.pickle"   \
    "embeddings_v_len60.pickle"    \
    "embeddings_v_len250.pickle"

#                     (accuracy, mean)  (accuracy, std)           (accuracy, <lambda_0>) metric_learning_embedding_name
#     pos model_type
#     4   linear                0.6010           0.0079  [0.591 0.598 0.601 0.602 0.613]       embeddings_v_base.pickle
#         xgb                   0.6107           0.0070  [0.606 0.608 0.609 0.609 0.623]       embeddings_v_base.pickle
#     5   linear                0.6246           0.0035  [0.620 0.622 0.624 0.628 0.628]      embeddings_v_hs512.pickle
#         xgb                   0.6265           0.0042  [0.622 0.624 0.625 0.629 0.633]      embeddings_v_hs512.pickle
#     6   linear                0.5200           0.0059  [0.514 0.517 0.518 0.520 0.530]   embeddings_v_epoch250.pickle
#         xgb                   0.5530           0.0073  [0.543 0.548 0.555 0.558 0.561]   embeddings_v_epoch250.pickle
#     7   linear                0.5003           0.0065  [0.492 0.497 0.500 0.503 0.509]       embeddings_v_bs32.pickle
#         xgb                   0.5807           0.0044  [0.575 0.577 0.582 0.584 0.585]       embeddings_v_bs32.pickle
#     8   linear                0.6224           0.0055  [0.618 0.619 0.621 0.624 0.631]      embeddings_v_bs128.pickle
#         xgb                   0.6224           0.0064  [0.614 0.620 0.622 0.627 0.630]      embeddings_v_bs128.pickle
#     9   linear                0.6242           0.0025  [0.620 0.623 0.625 0.626 0.627]      embeddings_v_bs256.pickle
#         xgb                   0.6229           0.0028  [0.621 0.621 0.623 0.623 0.628]      embeddings_v_bs256.pickle
#     10  linear                0.6192           0.0033  [0.617 0.617 0.618 0.619 0.625]     embeddings_v_len150.pickle
#         xgb                   0.6209           0.0037  [0.616 0.620 0.621 0.624 0.625]     embeddings_v_len150.pickle
#     11  linear                0.6195           0.0072  [0.609 0.616 0.622 0.623 0.628]      embeddings_v_len60.pickle
#         xgb                   0.6239           0.0067  [0.614 0.622 0.625 0.626 0.632]      embeddings_v_len60.pickle
#     12  linear                0.6188           0.0070  [0.609 0.617 0.620 0.622 0.627]     embeddings_v_len250.pickle
#         xgb                   0.6199           0.0059  [0.612 0.618 0.619 0.624 0.627]     embeddings_v_len250.pickle


python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_base.p"     output.path="../data/age-pred/finetuning_scores_v_base"       --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json
python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_hs512.p"    output.path="../data/age-pred/finetuning_scores_v_hs512"    params.rnn.hidden_size=512 --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json
python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_epoch250.p" output.path="../data/age-pred/finetuning_scores_v_epoch250"   --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json
python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_bs32.p"     output.path="../data/age-pred/finetuning_scores_v_bs32"       --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json
python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_bs128.p"    output.path="../data/age-pred/finetuning_scores_v_bs128"      --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json
python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_bs256.p"    output.path="../data/age-pred/finetuning_scores_v_bs256"      --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json
python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_len150.p"   output.path="../data/age-pred/finetuning_scores_v_len150"     --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json
python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_len60.p"    output.path="../data/age-pred/finetuning_scores_v_len60"      --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json
python -m scenario_age_pred fit_finetuning params.pretrained_model_path="models/age_pred_ml_model_v_len250.p"   output.path="../data/age-pred/finetuning_scores_v_len250"     --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json

python -m scenario_age_pred compare_approaches --pos 5 6 7 8 9 10 11 12 13 --target_score_file_names \
    "finetuning_scores_v_base"     \
    "finetuning_scores_v_hs512"    \
    "finetuning_scores_v_epoch250" \
    "finetuning_scores_v_bs32"     \
    "finetuning_scores_v_bs128"    \
    "finetuning_scores_v_bs256"    \
    "finetuning_scores_v_len150"   \
    "finetuning_scores_v_len60"    \
    "finetuning_scores_v_len250"

#                     (rocauc_score, mean)  (rocauc_score, std)       (rocauc_score, <lambda_0>)            target_scores_name
#     pos model_type
#     6   linear                    0.8711               0.0076  [0.865 0.866 0.867 0.875 0.883]      finetuning_scores_v_base
#         xgb                       0.8683               0.0078  [0.861 0.863 0.865 0.871 0.881]      finetuning_scores_v_base
#     7   linear                    0.8715               0.0058  [0.864 0.869 0.870 0.876 0.878]     finetuning_scores_v_hs512
#         xgb                       0.8685               0.0055  [0.859 0.869 0.869 0.872 0.873]     finetuning_scores_v_hs512
#     8   linear                    0.8711               0.0072  [0.865 0.866 0.868 0.876 0.881]  finetuning_scores_v_epoch250
#         xgb                       0.8678               0.0077  [0.861 0.863 0.864 0.873 0.879]  finetuning_scores_v_epoch250
#     9   linear                    0.8708               0.0101  [0.862 0.863 0.866 0.877 0.886]      finetuning_scores_v_bs32
#         xgb                       0.8671               0.0103  [0.857 0.860 0.864 0.873 0.882]      finetuning_scores_v_bs32
#     10  linear                    0.8713               0.0077  [0.865 0.866 0.867 0.877 0.882]     finetuning_scores_v_bs128
#         xgb                       0.8691               0.0065  [0.862 0.864 0.867 0.875 0.877]     finetuning_scores_v_bs128
#     11  linear                    0.8700               0.0056  [0.864 0.865 0.870 0.872 0.878]     finetuning_scores_v_bs256
#         xgb                       0.8662               0.0041  [0.860 0.864 0.868 0.869 0.870]     finetuning_scores_v_bs256
#     12  linear                    0.8573               0.0053  [0.851 0.854 0.856 0.861 0.864]    finetuning_scores_v_len150
#         xgb                       0.8526               0.0056  [0.846 0.847 0.854 0.857 0.859]    finetuning_scores_v_len150
#     13  linear                    0.8723               0.0062  [0.865 0.868 0.873 0.874 0.881]     finetuning_scores_v_len60
#         xgb                       0.8677               0.0075  [0.859 0.864 0.868 0.869 0.879]     finetuning_scores_v_len60
#     14  linear                    0.8705               0.0080  [0.861 0.865 0.870 0.876 0.881]    finetuning_scores_v_len250
#         xgb                       0.8669               0.0082  [0.858 0.861 0.866 0.872 0.878]    finetuning_scores_v_len250
