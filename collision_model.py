import pickle
import numpy as np


class CollisionModel:

    def __init__(self, model_path="models/best_model.pkl"):

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # If the pickle contains a dictionary
        if isinstance(model_data, dict):

            self.model = model_data.get("model")
            self.scaler = model_data.get("scaler", None)

        else:
            # If only the model was saved
            self.model = model_data
            self.scaler = None

    def _prep(self, features):
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return features

    def predict(self, features):
        features = self._prep(features)
        prediction = self.model.predict(features)[0]
        return prediction

    def predict_label(self, features):
        pred = self.predict(features)

        if pred == 0:
            return "SAFE"
        elif pred == 1:
            return "WARNING"
        else:
            return "DANGER"

    def predict_proba(self, features) -> float:
        """
        Continuous risk score in [0, 1].

        This is the key piece the old brake logic was missing: instead of
        the discrete label flipping SAFE/WARNING/DANGER on a knife-edge
        every frame (which made the brake bar jump), this gives a smooth
        confidence value that brake_predictor.py can blend in gradually
        and that adas_full_pipeline.py can EMA-smooth per track before
        it's even used.

        Uses predict_proba() if the underlying sklearn model supports it
        (RandomForest / GradientBoosting / LogisticRegression all do).
        If not, it falls back to a fixed score derived from the discrete
        label so this method always works regardless of model type.
        """
        features = self._prep(features)

        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(features)[0]
                classes = list(self.model.classes_)
                danger_p = float(proba[classes.index(2)]) if 2 in classes else 0.0
                warning_p = float(proba[classes.index(1)]) if 1 in classes else 0.0
                return max(0.0, min(1.0, danger_p + 0.5 * warning_p))
            except Exception:
                pass  # fall through to label-based fallback

        pred = int(self.predict(features))
        return {0: 0.0, 1: 0.5, 2: 1.0}.get(pred, 0.0)
