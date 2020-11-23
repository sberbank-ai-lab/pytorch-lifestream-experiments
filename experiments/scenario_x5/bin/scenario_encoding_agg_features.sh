# export SC_DEVICE="cuda"
# cd experiments/scenario_x5/

# preparing aggregated features dataset
python ../../agg_features_ts_preparation.py params.device="$SC_DEVICE" \
    --conf conf/dataset.hocon conf/agg_features_timestamps.json

# CPC features encoding
python ../../features_encoding.py params.device="$SC_DEVICE" --conf conf/agg_features_encoding.json

# preparing aggregated features embeddings
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/agg_features_timestamps.json

# preparing encoded aggregated features embeddings
python ../../ml_inference.py params.device="$SC_DEVICE" \
    model_path.model="models/agg_features_encoder.p" \
    output.path="data/agg_feat_encoded_embed" \
    --conf conf/dataset.hocon conf/agg_features_timestamps.json

# compare
python -m scenario_x5 compare_approaches --output_file "results/scenario_x5__encoding_agg_features.csv" \
    --n_workers 1 --models lgb --embedding_file_names \
    "agg_feat_embed.pickle"         \
    "agg_feat_encoded_embed.pickle"
