import numpy as np
import pandas as pd


def probabilistic_elo_init(players: pd.DataFrame, data: pd.DataFrame, base: np.float32 = 10.0, D: np.float32 = 400.0) -> pd.DataFrame:
    # The idea here is that we can use overall win-% as a basis for initial rating
    # A player with a P% win rate is expected to win against an average player P% of the time
    # This translates to have an expected score of P / 100 against a 1500 Elo player (the average)

    
    def get_base_boosts(player_ids) -> pd.Series:
        boosts = []
        for id in player_ids:
            # Thresholding to prevent explosion from small sample sizes
            
            try:

                if (wins + losses)[id] >= 10:
                    win_rate = np.min([0.85, np.max([0.15, win_rates[id]])])

                    boost = - D * np.log10((1 / win_rate) - 1)

                    boosts.append(boost)
                else:
                    boosts.append(0.0)
            except:
                boosts.append(0.0)

        # print(boosts)

        return pd.Series(boosts, player_ids)


    wins, losses = data['winner_id'].value_counts(), data['loser_id'].value_counts()
    win_rates = wins / (wins + losses)
    win_rates = win_rates.fillna(0.0)

    players['rating'] += get_base_boosts(players.index)

    return players