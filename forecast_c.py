"""
Публикация 2. Прогноз негибкой нагрузки дома сценария C на сутки вперёд.

План составляется в 17:00 на следующие 24 ч, поэтому используются только признаки,
известные к этому моменту: календарь, прогноз температуры (в расчёте — фактическая
температура как идеальный метеопрогноз), признак тёмного времени суток и лаги не короче 24 ч.
Обучение: 08.01–31.10.2025; проверка: 01.11–31.12.2025.
Результат: model_metrics.csv, forecast_test.csv.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")  # кириллица и символы в консоли Windows
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

df = pd.read_csv("house_c_hourly.csv", parse_dates=["timestamp"], index_col="timestamp")
y = df["inflex"]

X = pd.DataFrame(index=df.index)
X["hour"], X["dow"], X["off_day"] = df["hour"], df["dow"], df["off_day"]
X["temp"] = df["temp"]
X["temp_day"] = df["temp"].groupby(df.index.normalize()).transform("mean")
X["dark"] = df["dark"]
X["lag24"] = y.shift(24)
X["lag48"] = y.shift(48)
X["lag168"] = y.shift(168)
X["mean24_lag24"] = y.shift(24).rolling(24).mean()
X["same_hour_week_mean"] = (y.shift(24) + y.shift(48) + y.shift(72) + y.shift(96) + y.shift(120)
                            + y.shift(144) + y.shift(168)) / 7
X = X.dropna()
y = y.loc[X.index]

train = X.index < "2025-11-01"
Xtr, Xte, ytr, yte = X[train], X[~train], y[train], y[~train]

# линейной модели час и день недели подаются как категориальные признаки
lin = make_pipeline(
    ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["hour", "dow"])],
                      remainder="passthrough"),
    LinearRegression())
models = {
    "Линейная регрессия": lin,
    "Случайный лес": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.6,
                                           random_state=1, n_jobs=-1),
    "Градиентный бустинг": GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=4,
                                                     subsample=0.8, random_state=1),
}

pred = {"Наивный (неделю назад)": Xte["lag168"].values}
for name, m in models.items():
    m.fit(Xtr, ytr)
    pred[name] = np.clip(m.predict(Xte), 0.02, None)

hours = Xte["hour"].values
evening = (hours >= 17) & (hours <= 22)
day = Xte.index.normalize()
rows = {}
for name, p in pred.items():
    e = yte.values - p
    dmax_true = yte.groupby(day).max().values
    dmax_pred = pd.Series(p, index=Xte.index).groupby(day).max().values
    rows[name] = {
        "MAE, кВт": np.mean(np.abs(e)),
        "RMSE, кВт": np.sqrt(np.mean(e ** 2)),
        "MAPE, %": np.mean(np.abs(e) / yte.values) * 100,
        "MAE 17–23 ч, кВт": np.mean(np.abs(e[evening])),
        "Суточный максимум: прогноз/факт, %": dmax_pred.mean() / dmax_true.mean() * 100,
        "Доля часов с недооценкой > 0,5 кВт, %": np.mean(e > 0.5) * 100,
        "Средняя ошибка (смещение), кВт": np.mean(-e),
    }
metrics = pd.DataFrame(rows).T
metrics.to_csv("model_metrics.csv", float_format="%.3f")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(metrics.round(3))

out = pd.DataFrame({"fact": yte})
for name, p in pred.items():
    out[name] = p
out.to_csv("forecast_test.csv", float_format="%.4f")

gb = models["Градиентный бустинг"]
imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Важность признаков (бустинг):", imp.round(3).to_dict())
