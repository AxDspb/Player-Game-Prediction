# Player-Game-Prediction

Predicted Steam users game-playing likelihood (76% accuracy) and playtime duration (3-hour MSE) using collaborative filtering and latent factor models with regularized SVD matrix factorization and cosine similarity.

## Project Structure

- `PlayerGame.py`: Main script for training and prediction.
- `pairs_Played.csv`: Input data for play prediction.
- `pairs_Hours.csv`: Input data for time played prediction.
- `predictions_Played.csv`: Output predictions for play classification.
- `predictions_Hours.csv`: Output predictions for time played regression.

## Tasks

### Play Prediction

The model processes each user-game pair from `pairs_Played.csv`. For each pair:

- It checks whether the game is in the top 66% most played games.
- It computes the Cosine Similarity between the user and the game.

For each user, the games are sorted:

1. By whether the game is popular (in the top 66%).
2. By the average Cosine score.

### Time Played Prediction

This model uses a latent factor model with `alpha`, `betaU`, and `betaI`. TensorFlow is used to generate `gammaU` and `gammaI` using one latent factor.

- The TensorFlow `predict` function did not work as expected, so only the `gammaU` and `gammaI` variables it generated are used.
- Predictions are made using the formula: ```alpha + betaU + betaI + gammaU * gammaI```.