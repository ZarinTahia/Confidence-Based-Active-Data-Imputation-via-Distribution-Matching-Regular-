import torch
import torch.nn.functional as F
import numpy as np
from geomloss import SamplesLoss
from utils import nanmean, MAE, RMSE
import logging

class OTimputerDropOutNoise():
    def __init__(self,
                 eps=0.01,
                 lr=1e-2,
                 opt=torch.optim.RMSprop,
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=.9,
                 dropout_noise_scale=0.01):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.dropout_noise_scale = dropout_noise_scale
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")

    def fit_transform(self, X, verbose=True, report_interval=500, X_true=None, track_confidence=False, snapshot_interval=10, mask=None):
        X = X.clone()
        n, d = X.shape

        if self.batchsize > n // 2:
            self.batchsize = 2 ** int(np.log2(n // 2))
            if verbose:
                logging.info(f"Batchsize adjusted to {self.batchsize}.")

        if mask is None:
            mask = torch.isnan(X).double()
        else:
            mask = mask.double()

        # Initialize imputations
        imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        if verbose:
            logging.info(f"Starting OT with batchsize={self.batchsize}, epsilon={self.eps:.4f}")

        snapshots = [] if track_confidence else None

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)

        for i in range(self.niter):
            X_filled = X.clone()

            # Inject dropout-like noise into imputation BEFORE loss computation
            if self.dropout_noise_scale >= 0:
                noisy_imps = imps + self.dropout_noise_scale * torch.randn_like(imps)
            else:
                noisy_imps = imps

            X_filled[mask.bool()] = noisy_imps

            # Compute Sinkhorn loss
            loss = 0
            for _ in range(self.n_pairs):
                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)
                loss += self.sk(X_filled[idx1], X_filled[idx2])

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning("Nan or Inf detected in loss. Stopping.")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track snapshots of imputations for confidence estimation
            if track_confidence and (i % snapshot_interval == 0):
                snapshots.append(imps.detach().clone())

            if X_true is not None:
                maes[i] = MAE(X_filled, X_true, mask).item()
                rmses[i] = RMSE(X_filled, X_true, mask).item()

            if verbose and (i % report_interval == 0):
                msg = f"[Iter {i}] Loss: {loss.item() / self.n_pairs:.4f}"
                if X_true is not None:
                    msg += f" | MAE: {maes[i]:.4f} | RMSE: {rmses[i]:.4f}"
                logging.info(msg)

        # Final output
        if track_confidence and len(snapshots) > 1:
            snapshots_tensor = torch.stack(snapshots, dim=0)
            avg_imps = torch.mean(snapshots_tensor, dim=0)
            confidence_scores = torch.std(snapshots_tensor, dim=0)
        else:
            avg_imps = imps.detach()
            confidence_scores = None

        X_final = X.clone()
        X_final[mask.bool()] = avg_imps

        if X_true is not None:
            if track_confidence:
                return X_final, maes, rmses, confidence_scores
            else:
                return X_final, maes, rmses
        else:
            if track_confidence:
                return X_final, confidence_scores
            else:
                return X_final
