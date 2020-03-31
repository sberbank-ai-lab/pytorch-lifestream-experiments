export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python ../../metric_learning.py \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    params.train.split_strategy.cnt_min=200 \
    params.train.split_strategy.cnt_max=600 \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


export SC_SUFFIX="SplitRandom"
export SC_STRATEGY="SplitRandom"
python ../../metric_learning.py \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    params.train.max_seq_len=600 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_age_pred__subseq_smpl_strategy.csv" \
    --embedding_file_names \
    "emb__SplitRandom.pickle" \
    "emb__SampleRandom.pickle"


