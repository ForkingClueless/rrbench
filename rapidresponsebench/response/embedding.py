from .base import BaseResponse
from rapidresponsebench import INTERNAL_REFUSAL
import numpy as np
import pickle


class EmbeddingResponse(BaseResponse):
    def __init__(self, benign_prompts=None, llm=None):
        super().__init__(llm)
        from sklearn.linear_model import LogisticRegression
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.benign_prompts = [i.prompt for i in benign_prompts] or []
        self.jb_like_benign = []

        self.model = LogisticRegression(random_state=42)
        self.zero_shot = True

    def fit(self, jailbreak_prompts, jb_like_benign_prompts, benign_prompts):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        jailbreak_embeddings = self.encoder.encode(jailbreak_prompts)
        jb_like_benign_embeddings = self.encoder.encode(jb_like_benign_prompts)if jb_like_benign_prompts else []

        benign_embeddings = self.encoder.encode(benign_prompts)

        if len(jb_like_benign_embeddings) == 0:
            X = np.vstack((jailbreak_embeddings, benign_embeddings))
        else:
            X = np.vstack((jailbreak_embeddings, jb_like_benign_embeddings, benign_embeddings))

        y = np.array([1] * len(jailbreak_embeddings) + [0] * (len(jb_like_benign_embeddings) + len(benign_embeddings)))

        # Create sample weights
        sample_weights = np.ones(len(y))
        sample_weights[len(jailbreak_embeddings):len(jailbreak_embeddings)+len(jb_like_benign_embeddings)] = 0.5  # jb_like_benign
        sample_weights[len(jailbreak_embeddings)+len(jb_like_benign_embeddings):] = 5  # benign

        print(X.shape, y.shape)

        # Split the data into training and testing sets
        # train_test_split shuffles by default unless shuffle=False i s specified
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42
        )

        # Fit the model with sample weights
        self.model.fit(X_train, y_train, sample_weight=weights_train)

        # Evaluate the model's accuracy on the test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Benign', 'Jailbreak']))

        # Print the number of samples in each class
        print(f"\nTotal samples: {len(y)}")
        print(f"Jailbreak samples: {np.sum(y == 1)}")
        print(f"Benign samples: {np.sum(y == 0)}")

    def notify_benign(self, prompts):
        # self.jb_like_benign = [i.prompt for i in prompts]
        # print("notified benign", len(self.jb_like_benign))
        pass

    def notify(self, prompts):
        if prompts:
            jailbreak_prompts = [i.prompt for i in prompts]
            self.fit(jailbreak_prompts, self.jb_like_benign, self.benign_prompts)
            self.zero_shot = False

    def predict(self, prompts):
        embeddings = self.encoder.encode(prompts)
        return self.model.predict(embeddings)

    def query(self, prompts, *args, **kwargs):
        outputs = super().query(prompts, *args, **kwargs)

        if self.zero_shot:
            return outputs
        preds = self.predict([i.prompt for i in prompts])
        for i in range(len(preds)):
            if preds[i]:
                outputs[i] = INTERNAL_REFUSAL
        return outputs

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['encoder']
        state['model_state'] = pickle.dumps(self.model)
        del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = pickle.loads(state['model_state'])
        del self.__dict__['model_state']
