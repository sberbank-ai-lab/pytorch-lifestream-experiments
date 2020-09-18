for SC_HIDDEN_SIZE in 0064 0160 0480 0800
do
  export SC_SUFFIX="hidden_size__hs_${SC_HIDDEN_SIZE}"
  python ../../metric_learning.py \
      params.device="$SC_DEVICE" \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
  python ../../ml_inference.py \
      params.device="$SC_DEVICE" \
      model_path.model="models/x5_mlm__$SC_SUFFIX.p" \
      output.path="data/emb__$SC_SUFFIX" \
      --conf "conf/dataset.hocon" "conf/mles_params.json"
done

# Compare
python -m scenario_x5 compare_approaches --output_file "results/scenario_x5__hidden_size.csv" \
    --n_workers 3 --models lgb --embedding_file_names \
    "emb__hidden_size__hs_*.pickle"
