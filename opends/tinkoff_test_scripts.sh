# new embeddings
python metric_learning.py params.device="cuda:1" params.train.n_epoch=50 model_path.model="models/tinkoff_ml_model_v2.p" --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json
python ml_inference.py params.device="cuda:1" model_path.model="models/tinkoff_ml_model_v2.p" output.path="../data/tinkoff/embeddings_v2" --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json
python -m scenario_tinkoff train --name 'ml embeddings v2'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:1" --embedding_file_name "embeddings_v2.pickle"
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"

# hyper parameters
python -m scenario_tinkoff train --name 'ml embeddings opt 1'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:0" --optim_weight_decay 0.0001 0 0 0.0001
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"
python -m scenario_tinkoff train --name 'ml embeddings opt 2'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:0" --optim_weight_decay 0
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"
python -m scenario_tinkoff train --name 'ml embeddings opt 3'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:0" --optim_weight_decay 0.00001
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"


python -m scenario_tinkoff train --name 'ml embeddings opt 4'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:1" --optim_weight_decay 0.0001 0 0 0.0001 --optim_lr 0.015
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"
python -m scenario_tinkoff train --name 'ml embeddings opt 5'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:1" --optim_weight_decay 0 --optim_lr 0.015
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"
python -m scenario_tinkoff train --name 'ml embeddings opt 6'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:1" --optim_weight_decay 0.00001 --optim_lr 0.015
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"

python -m scenario_tinkoff train --name 'ml embeddings opt 7'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:1" --optim_weight_decay 0.0001 0 0 0.0001 --optim_lr 0.007
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"
python -m scenario_tinkoff train --name 'ml embeddings opt 8'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:1" --optim_weight_decay 0 --optim_lr 0.007
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"
python -m scenario_tinkoff train --name 'ml embeddings opt 9'  --use_embeddings --user_layers 1T --item_layers E1 --device "cuda:1" --optim_weight_decay 0.00001 --optim_lr 0.007
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"
