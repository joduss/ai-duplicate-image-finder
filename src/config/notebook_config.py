import os

from src.data.dataset.image_shape import ImageShape


class NotebookConfig:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_output_path = f"./output/{self.model_name}" if not os.getcwd().startswith(
        "/content") else f"/content/drive/MyDrive/Colab Notebooks/image_duplicate/output/{model_name}"

        self.tensorboard_log_dir = f"./output/{self.model_name}" if not os.getcwd().startswith(
        "/content") else f"/content/drive/MyDrive/Colab Notebooks/image_duplicate/output/tensorboard/{model_name}"

        self.image_shape = ImageShape(height=224, width=224, depth=3)

    @property
    def saved_model_path(self) -> str:
        """
        :return: Path to the model in the directory "$PROJECT_ROOT/models".
        """
        return f"./models/{self.model_name}" if not os.getcwd().startswith(
            "/content") else f"/content/drive/MyDrive/Colab Notebooks/image_duplicate/models/{self.model_name}"