from typing import Iterable

import numpy as np
import tensorflow.keras as k
import sklearn.metrics as skm
from matplotlib import pyplot as plt
from tensorflow.python.data import AUTOTUNE

from src.data.dataset.dataset import Dataset
from src.data.dataset.image_pair import ImagePair
from src.data.dataset.image_shape import ImageShape
from src.data.utility.tf_dataset_transformer import TfDatasetTransformer
from src.visualisations import image_plot as ipl

class Evaluation:

    def __init__(self, dataset: Dataset, model: k.Model, image_shape: ImageShape):
        self.dataset = dataset
        self.tf_dataset = TfDatasetTransformer(image_shape).transform_to_tf_dataset(dataset=dataset).batch(256).prefetch(AUTOTUNE)
        self.model = model
        self._predictions: np.ndarray or None = None


    def clear(self):
        self._predictions = None


    def _classify(self, workers=8):
        if self._predictions is None:
            self._predictions = self.model.predict(self.tf_dataset, workers=workers, max_queue_size=20, verbose=1)


    def print_f1_report(self, threshold=0.5, workers=8):
        self._classify(workers=workers)
        print(skm.classification_report(self.dataset.similarities(), self._predictions >= threshold, zero_division=0))


    def plot_roc(self, workers=8):
        self._classify(workers=workers)
        fpr, tpr, thresholds = skm.roc_curve(self.dataset.similarities(), self._predictions)
        roc_auc = skm.auc(fpr, tpr)
        display = skm.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="All")
        display.plot()
        plt.show()


    def print_separate_f1_reports(self, threshold=0.5, workers=8):
        self._classify(workers=workers)

        regular_true, regular_predictions, augmented_true, augmented_predictions = self._regular_augmented_plit()

        print("Regular pairs:")
        print(skm.classification_report(regular_true, regular_predictions >= threshold, zero_division=0))

        print("Augmented pairs:")
        print(skm.classification_report(augmented_true, augmented_predictions >= threshold, zero_division=0))


    def _regular_augmented_plit(self):
        idx_pairs_augmented = [(i, pair) for i, pair in enumerate(self.dataset.image_pairs) if pair.augmented]
        idx_pairs_regular = [(i, pair) for i, pair in enumerate(self.dataset.image_pairs) if not pair.augmented]

        idx_augm_pairs, augmented_pairs = zip(*idx_pairs_augmented)
        idx_regular_pairs, regular_pairs = zip(*idx_pairs_regular)
        idx_augm_pairs = list(idx_augm_pairs)
        idx_regular_pairs = list(idx_regular_pairs)

        augmented_predictions = self._predictions[idx_augm_pairs]
        regular_predictions = self._predictions[idx_regular_pairs]

        augmented_true = list(map(lambda pair: pair.similar, augmented_pairs))
        regular_true = list(map(lambda pair: pair.similar, regular_pairs))

        return regular_true, regular_predictions, augmented_true, augmented_predictions

    def show_images(self, range: Iterable, threshold=0.5, image_size_factor=1, workers=8):
        self._classify(workers=workers)

        similarities = self.dataset.similarities()

        image_pairs = []
        titles = []

        for i in range:
            image_pairs.append(self.dataset.image_pairs[i])
            similarity_true = similarities[i]
            similarity_pred = self._predictions[i]

            if similarity_true == int(similarity_pred >= threshold):
                titles.append(f"Different (sim={similarity_pred})" if similarity_true == 0 else f"Similar ({similarity_pred})")
            elif similarity_true == 0:
                titles.append(f"WRONG. Images are different. Pred sim: {similarity_pred}")
            else:
                titles.append(f"WRONG. Images are similar. Pred sim: {similarity_pred}")


        ipl.plot(image_pairs, titles=titles, fig_size_multiplier=image_size_factor)
