## Results 

Task                                              |    top_k_recall    |  Accuracy  |           comments             |           error rate           |
--------------------------------------------------| ------------------ | ---------- | -------------------------------| -------------------------------|
**mnist metric learning pretrain**                | **0.75**           |            |                                |
**cifar10 metric learning pretrain**              | **0.79**           |            |                                |
**cifar10 metric learning pretrain 2**            | **0.82**           |            | give better results for domysh.|
--------------------------------------------------|--------------------|------------|--------------------------------| -------------------------------|
**mnist domyshnik**                               |                    | **0.75**   |                                |             **0.1**            |
**mnist domyshnik with known critic**             |                    | **0.82**   | we use knowlege of environment |             **0.1**            |
**cifar10 domyshnik**                             |                    | **0.774**  | with unfreeze  metric learn mod|             **0.5**            |
**cifar10 domyshnik**                             |                    | **0.813**  | with unfreeze  metric learn mod(centroids)|  **0.5**            |
--------------------------------------------------|--------------------|------------|--------------------------------| -------------------------------|
**okko domyshnik**                                |                    | **0.33/0.344**  |                                |         **0.5/0.1**            |