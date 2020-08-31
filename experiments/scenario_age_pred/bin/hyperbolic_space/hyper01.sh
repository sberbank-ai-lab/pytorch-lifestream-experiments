export SC_SUFFIX="hyper01_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss   params.train.margin=0.5 \
    params.train.lr=0.002 \
    params.rnn.hidden_size=480 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100
do
    python ../../ml_inference.py \
        params.rnn.hidden_size=480 \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper02_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.002 \
    params.rnn.hidden_size=480 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100
do
    python ../../ml_inference.py \
        params.rnn.hidden_size=480 \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper03_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=4.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.002 \
    params.rnn.hidden_size=480 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100
do
    python ../../ml_inference.py \
        params.rnn.hidden_size=480 \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper04_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=4.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.001 \
    params.rnn.hidden_size=480 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100
do
    python ../../ml_inference.py \
        params.rnn.hidden_size=480 \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper05_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=1.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.002 \
    params.rnn.hidden_size=480 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100
do
    python ../../ml_inference.py \
        params.rnn.hidden_size=480 \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper06_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=0.5 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.002 \
    params.rnn.hidden_size=480 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100
do
    python ../../ml_inference.py \
        params.rnn.hidden_size=480 \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

python -m scenario_age_pred compare_approaches --n_workers 2 --models lgb \
    --embedding_file_names "mles_hyper06*_050.pickle" "mles_hyper06*_100.pickle"


lgb_embeds: mles_hyper01_base_050.pickle           0.6322  0.6280  0.6363 0.0033  [0.629 0.630 0.631 0.634 0.637]        0.6333  0.6292  0.6374 0.0033  [0.629 0.632 0.633 0.636 0.637]
lgb_embeds: mles_hyper02_poincare_050.pickle       0.6280  0.6220  0.6339 0.0048  [0.621 0.626 0.627 0.632 0.633]        0.6325  0.6252  0.6398 0.0059  [0.625 0.630 0.631 0.635 0.641]
lgb_embeds: mles_hyper03_poincare_050.pickle       0.6293  0.6243  0.6343 0.0041  [0.625 0.626 0.629 0.632 0.634]        0.6335  0.6288  0.6381 0.0037  [0.630 0.631 0.632 0.636 0.638]
lgb_embeds: mles_hyper04_poincare_050.pickle       0.6226  0.6151  0.6302 0.0061  [0.616 0.618 0.624 0.624 0.631]        0.6289  0.6270  0.6309 0.0016  [0.626 0.629 0.629 0.630 0.630]
lgb_embeds: mles_hyper05_poincare_050.pickle       0.6254  0.6222  0.6286 0.0026  [0.622 0.623 0.626 0.627 0.628]        0.6259  0.6188  0.6329 0.0057  [0.619 0.622 0.626 0.631 0.632]

lgb_embeds: mles_hyper01_base_100.pickle           0.6394  0.6298  0.6491 0.0078  [0.627 0.639 0.641 0.642 0.648]        0.6384  0.6369  0.6399 0.0012  [0.636 0.638 0.639 0.639 0.639]
lgb_embeds: mles_hyper02_poincare_100.pickle       0.6227  0.6147  0.6308 0.0065  [0.616 0.616 0.624 0.626 0.631]        0.6308  0.6270  0.6346 0.0030  [0.628 0.630 0.630 0.630 0.636]
lgb_embeds: mles_hyper03_poincare_100.pickle       0.6194  0.6144  0.6244 0.0040  [0.616 0.617 0.617 0.621 0.626]        0.6273  0.6247  0.6298 0.0021  [0.625 0.627 0.627 0.627 0.631]
lgb_embeds: mles_hyper04_poincare_100.pickle       0.6244  0.6200  0.6288 0.0035  [0.619 0.623 0.625 0.627 0.628]        0.6293  0.6248  0.6338 0.0036  [0.626 0.628 0.629 0.629 0.635]
lgb_embeds: mles_hyper05_poincare_100.pickle       0.6303  0.6268  0.6337 0.0028  [0.628 0.629 0.629 0.630 0.635]        0.6305  0.6275  0.6336 0.0025  [0.628 0.628 0.631 0.632 0.634]
