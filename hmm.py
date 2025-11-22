import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def adj_close(df, tic=None):
    if isinstance(df.columns, pd.MultiIndex):
        if tic and ('Adj Close', tic) in df.columns:
            return df[('Adj Close', tic)].dropna()
        try:
            s = df.xs('Adj Close', axis=1, level=0)
            return s.iloc[:,0].dropna() if isinstance(s, pd.DataFrame) else s.dropna()
        except:
            pass
    for c in ['Adj Close','Adj_Close','AdjClose','Close']:
        if c in df.columns:
            return df[c].dropna()
    raise KeyError("Adj close not found")

tic = "AAPL"
start = "2015-01-01"
end   = "2025-11-20"

data = yf.download(tic, start=start, end=end, progress=False)
if data.empty:
    raise RuntimeError("Download failed")

p0 = adj_close(data, tic)
p0 = p0.asfreq("B").ffill()

ret = np.log(p0).diff().dropna()
X = ret.values.reshape(-1,1)

sc = StandardScaler()
Xz = sc.fit_transform(X)

def fit(X, k):
    m = GaussianHMM(n_components=k, covariance_type="full",
                    n_iter=200, random_state=42)
    m.fit(X)
    return m

def bic(m, X):
    ll = m.score(X)
    k = m.n_components
    d = X.shape[1]
    t = k*(k-1)
    s = k-1
    c = k*(d*(d+1)/2)
    mu = k*d
    p = t + s + mu + c
    N = len(X)
    return -2*ll + p*np.log(N)

res = []
for k in range(2,6):
    m = fit(Xz, k)
    b = bic(m, Xz)
    res.append((k, m, b))
    print(f"k={k} BIC={b:.2f}")

model = min(res, key=lambda x: x[2])[1]
print("\nBest states:", model.n_components)

states = model.predict(Xz)
probs = model.predict_proba(Xz)

p = pd.Series(p0.loc[ret.index].values, index=ret.index)
r = pd.Series(ret.values, index=ret.index)
s = pd.Series(states, index=ret.index)
df = pd.DataFrame({"price": p, "ret": r, "state": s})

info = []
for st in range(model.n_components):
    idx = (states == st)
    info.append((st, ret.values[idx].mean(), ret.values[idx].std(), idx.sum()))
state_info = pd.DataFrame(info, columns=["state","mean","std","count"])
print("\nState info:\n", state_info)

print("\nTransition Matrix:\n", model.transmat_)

plt.figure(figsize=(14,6))
for st in range(model.n_components):
    m = (df.state == st)
    plt.plot(df.index[m], df.price[m], ".", ms=3, label=f"S{st}")
plt.legend()
plt.title(f"{tic} Price by State")
plt.show()

plt.figure(figsize=(14,4))
plt.plot(df.index, df.ret, lw=0.8)
for st in range(model.n_components):
    m = (df.state == st)
    plt.scatter(df.index[m], df.ret[m], s=8, label=f"S{st}")
plt.legend()
plt.title("Returns with States")
plt.show()

last = 200
plt.figure(figsize=(14,4))
plt.stackplot(df.index[-last:], probs[-last:].T,
              labels=[f"S{i}" for i in range(model.n_components)])
plt.legend()
plt.title("State Probabilities")
plt.show()

p_last = probs[-1]
p_next = p_last @ model.transmat_
print("\nLast posterior:", np.round(p_last,3))
print("Next-day prob:", np.round(p_next,3))

h = 5
A = model.transmat_
A_h = np.linalg.matrix_power(A, h)
p_h = p_last @ A_h
print(f"{h}-day prob:", np.round(p_h,3))

import joblib
joblib.dump({"model":model,"scaler":sc,"ticker":tic},
            f"hmm_{tic}_{model.n_components}states.pkl")
print("Model saved.")
