# LSTM encoder
export SC_SUFFIX="encoder_lstm"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.type="lstm" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Transformer encoder
export SC_SUFFIX="encoder_transf"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.model_type="transf" \
    params.train.batch_size=128 \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=4 \
    params.transf.input_size=128 \
    params.transf.dim_hidden=128 \
    params.transf.n_layers=4 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=128 \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_age_pred__encoder_types.csv" \
    --n_workers 2 --models lgb --embedding_file_names \
    "mles_embeddings.pickle"              \
    "emb__encoder_lstm.pickle"            \
    "emb__encoder_transf.pickle"
