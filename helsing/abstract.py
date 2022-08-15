from abc import ABC

import torch

from helsing.constants import BASE_DIR


class Model(ABC):

    def save(self, run_id, best=False):
        """
            Args:
                run_id: id of train run for which saving was donw
                best: if this save is best model of the run

            Returns:
                None
        """
        if best:
            path = BASE_DIR / run_id / "best-classifier.pt"
        else:
            path = BASE_DIR / run_id / "classifier.pt"
        state = {
            'state_dict': self.state_dict(),
            'params': {'config': self.config, 'num_classes': self.num_classes}
        }

        torch.save(state, path)
        print(f'Model saved at: {path}')

    @classmethod
    def load(cls, path):
        """
        Args:
            path: Path of model file. Check `.pt` files in run-id dir

        Returns:

        """
        data = torch.load(path, map_location=torch.device('cpu'))
        model = cls(**data['params'])
        model.load_state_dict(data['state_dict'])
        return model
