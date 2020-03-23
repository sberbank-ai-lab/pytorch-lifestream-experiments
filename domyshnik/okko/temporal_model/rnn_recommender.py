from typing import Optional

from pyspark.sql import DataFrame

from rnn_recommender_data import get_validation_data_loader, get_train_data_loader
from train import fit_model_classification, predict_model_classification
from model import get_model_calssification_with_features
from loss import KlLoss
import torch
import constants as cns
import pandas as pd


class RnnRecommender:

    def __init__(self):
        self.model = None
        self.features_map = None
        self.users_map = None

    @staticmethod
    def load():
        r = RnnRecommender()

        r.model = torch.load(f'./best_{cns.SAVE_WEIGHTS_NAME}')
        r.features_map = pd.read_csv(f'./features_map.csv')
        r.users_map = pd.read_csv(f'./users_map.csv')

        return r

    def fit(self,
            log: DataFrame,
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            path: Optional[str] = None) -> None:

        # remove deploy mode
        cns.DEPLOY_MODE = False

        data_loader, data_valid_loader = get_train_data_loader(log, user_features, item_features)

        model = get_model_calssification_with_features()

        loss = KlLoss()

        fit_model_classification(model, data_loader, data_valid_loader, loss)

    def predict(self,
                 k: int,
                 users: DataFrame,
                 items: DataFrame,
                 context: str,
                 log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 to_filter_seen_items: bool = True,
                 path: Optional[str] = None) -> DataFrame:

        # set deploy mode (not remove last state in rnn)
        cns.DEPLOY_MODE = True

        # predict only items from this list
        allowed_items = None
        if items is not None:
            allowed_items = torch.LongTensor(items.select('element_uid').distinct().toPandas()['element_uid'].values)

        data_loader = get_validation_data_loader(users, log, user_features, item_features)

        predictions = predict_model_classification(k, self.model, data_loader, to_filter_seen_items, allowed_items)

        # collect result
        df_predictions = pd.DataFrame({
            cns.USER_ID_COLUMN: predictions[:, 0].astype(int),
            cns.ELEMENT_ID_COLUMNS: predictions[:, 1].astype(int),
            'relevants': predictions[:, 2]
        })

        # mapping to original user and items codes
        df_predictions = pd.merge(df_predictions, self.features_map, on=cns.ELEMENT_ID_COLUMNS).reset_index()
        df_predictions = pd.merge(df_predictions, self.users_map, on=cns.USER_ID_COLUMN)
        df_predictions = df_predictions.drop(columns=[cns.USER_ID_COLUMN, cns.ELEMENT_ID_COLUMNS])
        df_predictions = df_predictions.rename(columns={
            f'orig_{cns.USER_ID_COLUMN}': cns.USER_ID_COLUMN,
            f'orig_{cns.ELEMENT_ID_COLUMNS}': cns.ELEMENT_ID_COLUMNS,
        })
        df_predictions = df_predictions[[cns.USER_ID_COLUMN, cns.ELEMENT_ID_COLUMNS, 'relevants']]
        df_predictions = df_predictions[df_predictions[cns.USER_ID_COLUMN] != -100]
        df_predictions.to_csv(f'./recommendation_relevants.csv')

        return df_predictions


# fit usage
#RnnRecommender().fit(None, None, None, None)

# predict usage
RnnRecommender.load().predict(4, None, None, None, None, None, None, None, None)