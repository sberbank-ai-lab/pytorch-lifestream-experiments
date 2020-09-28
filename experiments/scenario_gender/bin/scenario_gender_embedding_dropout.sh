for DROPOUT in 0.6 0.75 0.8 0.8
do
    python ../../metric_learning.py \
        params.device="$SC_DEVICE" \
        params.embeddings_dropout=$DROPOUT \
        model_path.model="models/gender_mlm__$DROPOUT.p" \
        --conf conf/dataset.hocon conf/mles_params.json
    python ../../ml_inference.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/gender_mlm__$DROPOUT.p" \
        output.path="data/emb_dropout_$DROPOUT" \
        --conf conf/dataset.hocon conf/mles_params.json

done

python -m scenario_gender compare_approaches --output_file "results/scenario_gender__embeddings_dropout.csv" \
    --n_workers 2 --models lgb --embedding_file_names \
    "emb_dropout_0.0.pickle"            \
    "emb_dropout_0.1.pickle"            \
    "emb_dropout_0.25.pickle"           \
    "emb_dropout_0.5.pickle"           \
    "emb_dropout_0.6.pickle"           \
    "emb_dropout_0.75.pickle"           \
    "emb_dropout_0.8.pickle"            \
    "emb_dropout_0.9.pickle"
