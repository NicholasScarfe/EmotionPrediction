# Predicting Emotional Trajectories in Mental Health Forums

Masters research project by Nicholas Scarfe (Adelaide University).

## Project description

This project asks whether the emotional direction of a mental health forum thread can be predicted while the thread is still going. Two things are predicted:

1. The average emotion of the rest of the comment section.
2. The average emotion of the original poster's own later replies.

The data is a private dataset from the Beyond Blue forums: nine sub-forums, around 4,000 posts and 200,000 comments. The dataset is not included in this repo.

Every post and comment is turned into a 43-dimensional emotion vector using the `borisn70/bert-43-multilabel-emotion-detection` model. A model then sees the post plus the first part of the comments (25%, 50% or 75%) and predicts the average emotion vector of the comments after that point.

Models tested: a mean baseline, ridge regression, an MLP, an LSTM and a transformer. Emotion vectors were also compared against SBERT text embeddings (`all-mpnet-base-v2`). Results are reported as a skill score (1 − model MSE / baseline MSE).

The main findings:

- The best model beats the mean baseline by a skill score of about 0.16. The gain is small, but it is consistent.
- Ridge regression on averaged emotion vectors does as well as the more complex models. Treating comments as an ordered sequence did not help.
- SBERT embeddings did not improve results and sometimes made them worse.
- The original post is a weak predictor of where the wider conversation goes, but a strong predictor of how its author will reply later.

The full write-up is in `Masters_RP_Final_Nscarfe.pdf`.

## Files

All notebooks were written for Google Colab and read and write data from Google Drive. Update the paths at the top of each notebook to match your own setup.

### `MasterRP_InitialVisualiser.ipynb`
Early exploration on a single sub-forum (anxiety). It cleans the posts and comments, swaps usernames for random IDs, and plots post and comment counts over time and word clouds. It also includes a test that splits one post into sentences and runs each sentence through the emotion model.

### `MasterRP_Predictor.ipynb`
The first prediction attempt. An LSTM predicts each comment's emotion vector from the comments before it. It is compared against a mean baseline and a persistence baseline (predict the previous vector again), on both the full 43 emotions and a simplified 3-way positive/neutral/negative version. This approach did no better than the mean, so the project moved to predicting thread averages instead.

### `MasterRP_EmotionConverter.ipynb`
Data cleaning and feature building for all nine sub-forums.
- Anonymises authors, fixes timestamps, converts emoji to text, and replaces usernames mentioned in comments with "someone".
- Splits each post and comment into sentences, gets an emotion vector per sentence, and averages them into one vector per message. Also creates an SBERT embedding for each message.
- Holds out 25% of threads per sub-forum as a test set.
- Builds the input features and targets for the 25%, 50% and 75% cutoffs and saves them as CSV and pickle files.

### `MasterRP_Final_Pipeline.ipynb`
The main experiments, run on the output of the EmotionConverter notebook. It is split into stages:
- **Stage 0:** mean and persistence baselines at each cutoff.
- **Stage 1:** ridge regression with emotion, SBERT and combined features, plus which emotions are easiest to predict.
- **Stage 2:** MLP with different feature sets removed.
- **Stage 3:** ridge vs MLP vs LSTM vs transformer on averaged inputs.
- **Stage 4:** sequential inputs vs averaged inputs.
- **Stage 5:** training on each sub-forum separately vs training on all of them together.
- **Stage 6:** comparing the 25%, 50% and 75% cutoffs.
- **Stage 7:** post only vs comments only vs both.

## Running order

1. `MasterRP_EmotionConverter.ipynb` to clean the data and build features.
2. `MasterRP_Final_Pipeline.ipynb` to run the experiments.

The other two notebooks are earlier exploratory work and are not needed to reproduce the final results.

## Requirements

Python with `pandas`, `numpy`, `torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `emoji`, `nltk`, `matplotlib`, `seaborn` and `wordcloud`. A GPU is recommended for the emotion conversion step.
