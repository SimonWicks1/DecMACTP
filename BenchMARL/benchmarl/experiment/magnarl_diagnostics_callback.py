import torch
from tensordict import TensorDict
from benchmarl.experiment.callback import Callback


class MagnarlDiagnosticsCallback(Callback):
    """
    Log MAGNARL internal diagnostics from:
      1) collection batches
      2) training batches
      3) evaluation rollouts

    Assumes the policy writes diagnostics under:
      (group, "diag", <key>)
      (group, "train_diag", <key>)   # optional
    """

    def __init__(self, group: str = "agents", log_every_collection: bool = True):
        super().__init__()
        self.group = group
        self.log_every_collection = log_every_collection

    @staticmethod
    def _to_scalar(x):
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.detach().to(torch.float)
        if x.numel() == 0:
            return None
        return x.mean().item()

    @staticmethod
    def _reduce_diag_td(diag_td, prefix: str):
        out = {}
        if diag_td is None:
            return out
        for key, value in diag_td.items():
            if torch.is_tensor(value):
                val = value.detach().to(torch.float)
                if val.numel() > 0:
                    out[f"{prefix}/{key}"] = val.mean().item()
        return out

    def _on_batch_collected(self, batch):
        """
        Collection-time logging.
        This is ideal for policy internal diagnostics produced during rollout.
        """
        if not self.log_every_collection:
            return

        diag_td = batch.get((self.group, "diag"), None)
        to_log = self._reduce_diag_td(diag_td, f"collection/{self.group}/diag")
        if to_log:
            self.experiment.logger.log(
                to_log, step=self.experiment.n_iters_performed
            )

    def _on_train_step(self, subdata, group):
        """
        Training-time logging.
        The return value is merged into training_td by Experiment._optimizer_loop(),
        and then logger.log_training() will record it automatically.
        """
        if group != self.group:
            return None

        out = {}

        # 1) optional explicit train diagnostics written by the model/loss
        train_diag_td = subdata.get((group, "train_diag"), None)
        if train_diag_td is not None:
            for key, value in train_diag_td.items():
                if torch.is_tensor(value):
                    value = value.detach().to(torch.float)
                    if value.numel() > 0:
                        out[f"diag_{key}"] = value.mean()

        # 2) also log rollout diagnostics if they survive into subdata
        diag_td = subdata.get((group, "diag"), None)
        if diag_td is not None:
            for key, value in diag_td.items():
                if torch.is_tensor(value):
                    value = value.detach().to(torch.float)
                    if value.numel() > 0:
                        out[f"diag_rollout_{key}"] = value.mean()

        if not out:
            return None
        return TensorDict(out, batch_size=[])

    def _on_evaluation_end(self, rollouts):
        """
        Evaluation-time logging from the rollout tensordicts.
        logger.log_evaluation() does not automatically scan custom diag keys,
        so we do it here.
        """
        acc = {}

        for td in rollouts:
            diag_td = td.get((self.group, "diag"), None)
            if diag_td is None:
                continue
            for key, value in diag_td.items():
                if not torch.is_tensor(value):
                    continue
                value = value.detach().to(torch.float)
                if value.numel() == 0:
                    continue
                acc.setdefault(key, []).append(value.mean())

        if not acc:
            return

        to_log = {}
        for key, values in acc.items():
            stacked = torch.stack(values)
            to_log[f"eval/{self.group}/diag/{key}_mean"] = stacked.mean().item()
            to_log[f"eval/{self.group}/diag/{key}_min"] = stacked.min().item()
            to_log[f"eval/{self.group}/diag/{key}_max"] = stacked.max().item()

        self.experiment.logger.log(to_log, step=self.experiment.n_iters_performed)