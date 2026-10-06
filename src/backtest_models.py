"""Walk-forward models for the backtester: each block of decisions is scored by models fitted only on bars whose labels were already
known at the block's first bar (the last `horizon` rows are purged), and the block is predicted in one batch."""


class WalkForwardModels:
    def __init__(self, X, y_long, y_short, prices, train_bars, every, horizon, fit_fn):
        self.X, self.y = X, {"long": y_long, "short": y_short}
        self.prices, self.horizon, self.fit_fn = prices, int(horizon), fit_fn
        self.train_bars, self.every = int(train_bars), int(every)
        self.start = self.train_bars
        self.fits = []                      # (retrain row, first train row, train end exclusive)
        self._ens = {"long": None, "short": None}
        self._block_start = None
        self._block = None

    def _fit_block(self, row):
        first, end = row - self.train_bars, row - self.horizon + 1   # row - horizon is the last label known at `row`
        stop = min(row + self.every, len(self.X))
        block = {}
        for side in ("long", "short"):
            ens = self.fit_fn(self.X.iloc[first:end], self.y[side].iloc[first:end], self.prices.iloc[first:end], side, self._ens[side])
            self._ens[side] = ens
            block[side] = (ens.predict_proba(self.X.iloc[row:stop]), float(ens.ensemble_cv_auc_))
        self.fits.append((row, first, end))
        self._block_start, self._block = row, block

    def probs(self, i):
        """(p_long, p_short, auc_long, auc_short) for decision row `i`, from the model trained for its block."""
        if i < self.start:
            raise ValueError(f"row {i} is inside the first training window (starts at {self.start})")
        row = self.start + ((i - self.start) // self.every) * self.every
        if row != self._block_start:
            self._fit_block(row)
        (pl, auc_l), (ps, auc_s) = self._block["long"], self._block["short"]
        t = self.X.index[i]
        return float(pl.loc[t]), float(ps.loc[t]), auc_l, auc_s


def make_fit_fn(cfg, sym, model_params, min_improvement):
    """The production fit: a fresh `Ensemble` the first time, then `safe_retrain_ensemble` (keeps the old model unless the new one
    improves by `min_improvement`, the live gate). `dry_run=True`: nothing is written to `models/`."""
    from src.ensemble import Ensemble
    from src.strategy_ml import MIN_SAMPLES_FOR_FIT
    from src.utils import safe_retrain_ensemble

    def fit_fn(Xt, yt, pt, side, previous):
        if len(Xt) < MIN_SAMPLES_FOR_FIT:   # the Ensemble would skip the fit with a warning and then predict 0.5 for every bar
            raise ValueError(f"{len(Xt)} training rows are below the {MIN_SAMPLES_FOR_FIT} a model needs: raise backtesting.train_bars")
        if previous is None:
            ens = Ensemble(cfg, model_params=model_params)
            ens.fit(Xt, yt, prices=pt, model_type=side)
            return ens
        return safe_retrain_ensemble(cfg, sym, previous, Xt, yt, pt, dry_run=True, model_type=side, model_params=model_params,
                                     min_improvement=min_improvement)

    return fit_fn
