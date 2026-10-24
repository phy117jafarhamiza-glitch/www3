# -*- coding: utf-8 -*-
"""
شبكة عصبية التفافية أحادية البعد (1D-CNN) مكتوبة بـ NumPy فقط، مع الانتشار
الأمامي والعكسي ومُحسِّن Adam. الهدف: تدريب وتشغيل EMG-Net في أي بيئة (حتى
Streamlit Cloud المجاني) دون الحاجة إلى TensorFlow أو PyTorch.

البنية الافتراضية (حسب الأطروحة):
  Conv(1→8,k=7)+ReLU+MaxPool2 → Conv(8→16,k=5)+ReLU+MaxPool2 →
  Conv(16→32,k=3)+ReLU → GlobalAvgPool → Dense(32→16)+ReLU → Dense(16→C)+Softmax
"""
import numpy as np


def _he(rng, shape, fan_in):
    return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)


class Conv1D:
    def __init__(self, cin, cout, k, rng):
        self.k, self.cin, self.cout = k, cin, cout
        self.W = _he(rng, (k * cin, cout), k * cin)
        self.b = np.zeros(cout)

    def forward(self, x):                       # x: (N, L, cin)
        N, L, _ = x.shape
        pl = (self.k - 1) // 2
        pr = self.k - 1 - pl
        xp = np.pad(x, ((0, 0), (pl, pr), (0, 0)))
        self.cols = np.concatenate([xp[:, i:i + L, :] for i in range(self.k)], axis=2)
        self.in_shape, self.pl, self.pr = x.shape, pl, pr
        return self.cols @ self.W + self.b

    def backward(self, dout):                   # dout: (N, L, cout)
        N, L, _ = dout.shape
        self.dW = np.einsum('nlc,nlo->co', self.cols, dout)
        self.db = dout.sum(axis=(0, 1))
        dcols = dout @ self.W.T                 # (N, L, k*cin)
        dxp = np.zeros((N, L + self.k - 1, self.cin))
        for i in range(self.k):
            dxp[:, i:i + L, :] += dcols[:, :, i * self.cin:(i + 1) * self.cin]
        return dxp[:, self.pl:self.pl + L, :]

    params = property(lambda s: [s.W, s.b])
    grads = property(lambda s: [s.dW, s.db])


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, d):
        return d * self.mask
    params = property(lambda s: [])
    grads = property(lambda s: [])


class MaxPool1D:
    def forward(self, x):                       # pool=2
        N, L, C = x.shape
        self.L = L
        L2 = L // 2
        xr = x[:, :L2 * 2, :].reshape(N, L2, 2, C)
        self.arg = xr.argmax(axis=2)
        return xr.max(axis=2)

    def backward(self, d):
        N, L2, C = d.shape
        out = np.zeros((N, L2, 2, C))
        idx = self.arg[:, :, None, :]
        np.put_along_axis(out, idx, d[:, :, None, :], axis=2)
        res = out.reshape(N, L2 * 2, C)
        if self.L > L2 * 2:
            res = np.concatenate([res, np.zeros((N, self.L - L2 * 2, C))], axis=1)
        return res
    params = property(lambda s: [])
    grads = property(lambda s: [])


class GlobalAvgPool:
    def forward(self, x):
        self.L = x.shape[1]
        return x.mean(axis=1)

    def backward(self, d):
        return np.repeat(d[:, None, :], self.L, axis=1) / self.L
    params = property(lambda s: [])
    grads = property(lambda s: [])


class Dense:
    def __init__(self, nin, nout, rng):
        self.W = _he(rng, (nin, nout), nin)
        self.b = np.zeros(nout)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.dW = self.x.T @ d
        self.db = d.sum(axis=0)
        return d @ self.W.T
    params = property(lambda s: [s.W, s.b])
    grads = property(lambda s: [s.dW, s.db])


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class EMGNet1DCNN:
    """شبكة CNN أحادية البعد لتصنيف نوافذ EMG إلى n_classes فئة."""

    def __init__(self, input_len=256, n_classes=3, seed=0, channels=(8, 16, 32), hidden=16):
        rng = np.random.default_rng(seed)
        self.input_len, self.n_classes = input_len, n_classes
        self.channels, self.hidden = tuple(channels), hidden
        c1, c2, c3 = channels
        self.layers = [
            Conv1D(1, c1, 7, rng), ReLU(), MaxPool1D(),
            Conv1D(c1, c2, 5, rng), ReLU(), MaxPool1D(),
            Conv1D(c2, c3, 3, rng), ReLU(), GlobalAvgPool(),
            Dense(c3, hidden, rng), ReLU(),
            Dense(hidden, n_classes, rng),
        ]
        self._adam = None

    # ------------------------------------------------------------ forward
    def _forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[:, :, None]
        out = []
        for i in range(0, len(X), 256):
            out.append(softmax(self._forward(X[i:i + 256])))
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # ----------------------------------------------------------- training
    def _all_params(self):
        ps = []
        for l in self.layers:
            ps.extend(l.params)
        return ps

    def _all_grads(self):
        gs = []
        for l in self.layers:
            gs.extend(l.grads)
        return gs

    def loss_and_backward(self, X, y, class_weights=None):
        logits = self._forward(X)
        p = softmax(logits)
        N = len(y)
        w = np.ones(N) if class_weights is None else np.asarray(class_weights)[y]
        loss = float(-(w * np.log(p[np.arange(N), y] + 1e-12)).sum() / w.sum())
        d = p.copy()
        d[np.arange(N), y] -= 1.0
        d = d * (w[:, None] / w.sum())
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return loss

    def _adam_step(self, lr, wd, t, b1=0.9, b2=0.999, eps=1e-8):
        params, grads = self._all_params(), self._all_grads()
        if self._adam is None:
            self._adam = ([np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params])
        m, v = self._adam
        for i, (p, g) in enumerate(zip(params, grads)):
            if p.ndim > 1:
                g = g + wd * p                  # L2 على الأوزان فقط
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** t)
            vh = v[i] / (1 - b2 ** t)
            p -= lr * mh / (np.sqrt(vh) + eps)  # تحديث في المكان

    @staticmethod
    def _augment(xb, rng):
        n = len(xb)
        xb = xb * rng.uniform(0.92, 1.08, size=(n, 1, 1))
        xb = xb + rng.normal(0, 0.01 * (np.abs(xb).mean() + 1e-9), size=xb.shape)
        shifts = rng.integers(-8, 9, size=n)
        for i, s in enumerate(shifts):
            xb[i] = np.roll(xb[i], s, axis=0)
        return xb

    def fit(self, X, y, epochs=25, batch_size=32, lr=3e-3, weight_decay=1e-4,
            class_weight='balanced', augment=True, seed=0, verbose=False, val=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[:, :, None]
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(seed)
        cw = None
        if class_weight == 'balanced':
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            counts[counts == 0] = 1.0
            cw = len(y) / (self.n_classes * counts)
        history, t = [], 0
        for ep in range(epochs):
            order = rng.permutation(len(X))
            ep_loss = 0.0
            for i in range(0, len(X), batch_size):
                idx = order[i:i + batch_size]
                xb = X[idx].copy()
                if augment:
                    xb = self._augment(xb, rng)
                t += 1
                ep_loss += self.loss_and_backward(xb, y[idx], cw) * len(idx)
                self._adam_step(lr, weight_decay, t)
            rec = {'epoch': ep + 1, 'loss': ep_loss / len(X)}
            if val is not None:
                Xv, yv = val
                rec['val_acc'] = float((self.predict(Xv) == np.asarray(yv)).mean())
            history.append(rec)
            if verbose:
                print(rec)
        return history

    # ---------------------------------------------------------- save/load
    def save(self, path):
        arrs = {f'p{i}': p for i, p in enumerate(self._all_params())}
        np.savez(path, input_len=self.input_len, n_classes=self.n_classes,
                 channels=np.array(self.channels), hidden=self.hidden, **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        net = cls(int(z['input_len']), int(z['n_classes']), channels=tuple(int(c) for c in z['channels']),
                  hidden=int(z['hidden']))
        for i, p in enumerate(net._all_params()):
            p[...] = z[f'p{i}']
        return net
