import pandas as pd
import nltk
from collections import Counter
import joblib
from src.utils import get_logger

logger = get_logger("Preprocessing")

class SalaryPreprocessor:
    """
    Data cleaning, dictionary building, and data transformation.
    """
    def __init__(self, config):
        self.config = config
        self.cat_to_id_dict = {}
        self.word_to_id = {}
        self.vocab_size_list = []
        self.is_fitted = False

    def fit(self, df):
        logger.info("Preparing categorical and text features...")

        # 1. Categorical features
        for col in self.config['data']['cat_features']:
            unique_values = df[col].fillna("Missing").astype(str).unique()
            mapping = {val: i + 1 for i, val in enumerate(unique_values)}
            self.cat_to_id_dict[col] = mapping
            self.vocab_size_list.append(len(mapping) + 1)
            logger.debug(f"Column '{col}' fitted with {len(mapping)} unique values.")

        # 2. Text Vocabulary
        text_col = self.config['data']['text_features'][0]
        tokens = []
        for text in df[text_col]:
            tokens.extend(nltk.word_tokenize(str(text).lower()))

        counts = Counter(tokens)
        min_freq = self.config['preprocessing']['min_freq']
        vocabulary = sorted([w for w, c in counts.items() if c >= min_freq])

        self.word_to_id = {word: i + 2 for i, word in enumerate(vocabulary)}
        self.word_to_id['<PAD>'] = 0
        self.word_to_id['<UNK>'] = 1

        logger.debug(f"Vocabulary created. Total words: {len(self.word_to_id)}")
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            logger.error("Transformation attempted before fitting!")
            raise ValueError("Preprocessor must be fitted before transformation.")

        logger.info("Transforming raw data...")
        
        # Categorical processing
        df_transformed = pd.DataFrame()
        for col in self.config['data']['cat_features']:
            df_transformed[col] = df[col].fillna("Missing").astype(str).map(
                lambda x: self.cat_to_id_dict.get(col, {}).get(x, 0)
            )

        # Text processing
        text_col = self.config['data']['text_features'][0]
        max_len = self.config['preprocessing']['max_len']
        
        sequences = []
        for text in df[text_col]:
            words = nltk.word_tokenize(str(text).lower())
            seq = [self.word_to_id.get(w, self.word_to_id['<UNK>']) for w in words]
            
            # Padding / Truncating
            if len(seq) < max_len:
                seq += [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            sequences.append(seq)

        return df_transformed, sequences

    def save(self, path):
        """Save the fitted preprocessor to a file."""
        joblib.dump(self, path)
        logger.info(f"Preprocessor artifacts saved to: {path}")

    @staticmethod
    def load(path):
        """Loads a preprocessor from a file."""
        return joblib.load(path)