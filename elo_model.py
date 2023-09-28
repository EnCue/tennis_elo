import pandas as pd
import numpy as np
from typing import Callable
from sklearn.metrics import log_loss
from data_forms import TrainingData

class EloModel:
    
    def __init__(self, initial_data: TrainingData, base_rating: np.float32, base: np.float32 = 10.0, D: np.float32 = 400.0, **kwargs):
        self.base_rating = base_rating

        init_heuristics = kwargs.get('init_heuristics', None)
        self.ratings = self.init_ratings(initial_data.data)

        if init_heuristics is not None:
            self.ratings = init_heuristics(self.ratings, initial_data.partitioned_data['Training'])

        self.base = base
        self.D = D
    

    # Initialize ratings as generic DataFrame
    def init_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        winner_ids = df[['winner_id', 'winner_name']].drop_duplicates().rename(columns={'winner_id': 'id', 'winner_name': 'name'})
        loser_ids = df[['loser_id', 'loser_name']].drop_duplicates().rename(columns={'loser_id': 'id', 'loser_name': 'name'})
        
        # player_ratings = pd.concat([winner_ids, loser_ids]).set_index('id')

        # For some reason concat doesn't merge correctly so dupes need to be dropped manually
        # player_ratings.drop_duplicates(inplace=True)

        player_ratings = pd.concat([winner_ids, loser_ids]).drop_duplicates().set_index('id')


        # Initializing rating column with base rating
        player_ratings['rating'] = self.base_rating
        # Initializing games-played column with 0
        player_ratings['games'] = 0

        # Apply custom initialization heuristics if specified
        # if heuristics is not None:
        #     player_ratings = heuristics(player_ratings, df)

        return player_ratings


    # Generic K factor function
    def K_factor(self, rating: np.float32) -> np.float32:
        K = 32

        if rating > 2100 and rating < 2400:
            K = 24
        elif rating > 2400:
            K = 16
        
        return K


    # Update ratings with new data
    def update_elo(self, training_data: pd.DataFrame):

        # Extract game info from row
        def get_score_info(match_df: pd.DataFrame) -> dict:
            score = match_df['score']
            bestof = match_df['best_of']

            if "RET" in score:
                # print("RET found")
                # THROW ERROR ?
                return None
            
            rounds_played = len(score.split())
            
            winner_score = np.ceil(bestof / 2)
            loser_score = rounds_played - winner_score

            score_info = {
                'Winner': winner_score,
                'Loser': loser_score,
                'Total': bestof
            }

            return score_info

        def calc_dElo(r_a: np.float32, r_b: np.float32, score: dict) -> tuple[np.float32, np.float32]:

            E_a = get_EV(r_a, r_b)
            E_b = 1 - E_a

            S_a = score['Winner'] / score['Total']
            S_b = 1 - S_a

            d_a = self.K_factor(r_a) * (S_a - E_a)
            d_b = self.K_factor(r_b) * (S_b - E_b)

            # r_a += d_a
            # r_b += d_b

            return (d_a, d_b)

        def get_EV(r_a: np.float32, r_b: np.float32, base=10, D=400) -> np.float32:
            # Implements Elo formula and returns expected score for winner
            Q_a = base ** (r_a / D)
            Q_b = base ** (r_b / D)

            E_a = Q_a / (Q_a + Q_b)

            return E_a
        

        RET_COUNT = 0
        # SORT ROWS BY "TOURNEY_DATE"
        for _, row in training_data.iterrows():
            winner_id, loser_id = row['winner_id'], row['loser_id']

            match_df = row[['score', 'best_of']]
            score_info = get_score_info(match_df)

            # IF RET -> SKIP ROW
            if not score_info:
                RET_COUNT += 1
                continue

            # Getting current ratings
            r_winner, r_loser = self.ratings.at[winner_id, 'rating'], self.ratings.at[loser_id, 'rating']

            # Calculating change in Elo rating
            d_winner, d_loser = calc_dElo(r_winner, r_loser, score_info)


            # Updating Elo rating
            self.ratings.at[winner_id, 'rating'] += d_winner
            self.ratings.at[loser_id, 'rating'] += d_loser

            # Updating games played
            self.ratings.at[winner_id, 'games'] += 1
            self.ratings.at[loser_id, 'games'] += 1


    # Get EV for player IDs
    def getEV(self, id_A, id_B) -> tuple:
        r_a, r_b = self.ratings.at[id_A, 'rating'], self.ratings.at[id_B, 'rating']
        Q_a = self.base ** (r_a / self.D)
        Q_b = self.base ** (r_b / self.D)

        E_a = Q_a / (Q_a + Q_b)

        return (E_a, 1 - E_a)
    
    # Makes current-state Elo predictions for given matches
    def predict(self, matches: pd.DataFrame) -> pd.DataFrame:

        preds = matches.apply(lambda row: self.getEV(row['id_A'], row['id_B']), axis=1, result_type='expand')

        return preds




# Train and validate predictions on test set // Returns CE of predictions
def evaluate_EloModel(dataset: TrainingData, **kwargs) -> dict:
    train_df, test_df = dataset.partitioned_data['Training'], dataset.partitioned_data['Testing']

    # Model must be initialized on complete dataset to ensure support for new players in test data
    init_heuristics = kwargs.get('init_heuristics', None)
    model = EloModel(dataset, 1500.0, init_heuristics = init_heuristics)
    # Model is only trained on training set
    model.update_elo(train_df)

    CE = 0
    N = 0
    # tourney_preds = {}
    preds = pd.DataFrame()
    test_ids = test_df['tourney_id'].unique()
    for tourney_id in test_ids:

        tourney = test_df.loc[test_df['tourney_id'] == tourney_id]

        matches = tourney.rename(columns={'winner_id': 'id_A', 'loser_id': 'id_B'})

        tourney_preds = model.predict(matches)
        preds = pd.concat([preds, tourney_preds])
        

        # update Elo from actual results
        model.update_elo(tourney)
    
    probs_np = preds.to_numpy()
    N = probs_np.shape[0]
    labels_np = np.tile([1, 0], (N, 1))
    CE = log_loss(labels_np, probs_np)

    preds_np = np.argmax(probs_np, axis=1)
    accuracy = (preds_np == 0).sum() / N

    brier_score = np.sum(( probs_np[:, 0] - 1.0 ) ** 2) / N

    metrics = {
        'N': N,
        'CE': CE,
        'Accuracy': accuracy,
        'BS': brier_score
    }
    
    return metrics




