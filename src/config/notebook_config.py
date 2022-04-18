import os

from src.data.dataset.image_shape import ImageShape


class NotebookConfig:

    def __init__(self, model_name: str, dataset_version: str = None):

        self.dataset_version = ""
        if dataset_version is not None:
            self.dataset_version = f"_{dataset_version}"

        self.model_name = model_name
        self.model_output_path = self._make_path(f"output/{self.model_name}{self.dataset_version}")

        self.tensorboard_log_dir = self._make_path(f"output/tensorboard/{model_name}{self.dataset_version}")

        self.image_shape = ImageShape(height=224, width=224, depth=3)

    @property
    def saved_model_path(self) -> str:
        """
        :return: Path to the read-only model in the directory "$PROJECT_ROOT/models".
        """
        return self._make_path(f"models/tensorflow/{self.model_name}{self.dataset_version}")


    @property
    def image_size(self) -> tuple:
        return self.image_shape.size


    @property
    def core_ml_path(self) -> str:
        return self._make_path(f"models/coreml/{self.model_name}.mlmodel")


    def summary(self) -> str:
        string_summary = f"Model: {self.model_name}\n"
        string_summary += f"Output: {self.model_output_path}\n"
        string_summary += f"Output for Tensorboard: {self.tensorboard_log_dir}\n"
        string_summary += f"Saved model path: {self.saved_model_path}\n"
        string_summary += "\n"
        string_summary += f"Image shape: {self.image_shape}\n"

        return string_summary


    def _make_path(self, relative_path: str) -> str:
        if relative_path.startswith("/"):
            relative_path = relative_path[1:]

        if relative_path.startswith("./"):
            relative_path = relative_path[2:]

        path = f"./{relative_path}" if not os.getcwd().startswith(
            "/content") else f"/content/drive/MyDrive/Colab Notebooks/image_duplicate/{relative_path}"

        return path[:-1] if path.endswith("/") else path
