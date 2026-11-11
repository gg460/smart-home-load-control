"""
Публикация 2. Алгоритм управления гибкими нагрузками дома сценария C по прогнозу
негибкой нагрузки и трёхзонному тарифу (постановление Совмина РБ № 93 от 25.02.2026,
дом с электроплитой): ночь 23–06 ч — 0,1549; пик 17–23 ч — 0,4646; остальное время —
0,1807 BYN/кВт·ч; в выходные и праздники — ставка ночной зоны весь день.

Каждое гибкое задание планируется в момент появления (онлайн) в окне [release, deadline):
  стоимость часа t для добавки δ  =  c_t·δ + μ·[(L_t+δ)² − L_t²] + ν·Δ(ступеньки) + штраф за L_t > P_lim,
где L_t — плановая нагрузка дома (прогноз негибкой + уже запланированные гибкие).
Неделимые программы (стирка+сушка, посудомойка) — перебор времени старта;
делимые (электромобиль, бойлер) — пошаговое «заполнение провала» квантами 0,1 кВт·ч.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")  # кириллица и символы в консоли Windows
import numpy as np
import pandas as pd

MU, NU, P_LIM, Q = 0.012, 0.012, 7.0, 0.1
TEST_START, TEST_END = "2025-11-01", "2026-01-01"

df = pd.read_csv("house_c_hourly.csv", parse_dates=["timestamp"], index_col="timestamp")
jobs = pd.read_csv("house_c_jobs.csv", keep_default_na=False)
for col in ["release", "deadline"]:
    jobs[col] = jobs[col].astype(int)
for col in ["energy", "pmax"]:
    jobs[col] = pd.to_numeric(jobs[col], errors="coerce")
fc = pd.read_csv("forecast_test.csv", parse_dates=["timestamp"], index_col="timestamp")

t0 = df.index.get_loc(pd.Timestamp(TEST_START))
idx = df.index[t0:]
n = len(idx)
hour = df["hour"].values[t0:]
off = df["off_day"].values[t0:].astype(bool)


def tariff3(h, off_day):
    if off_day or h >= 23 or h < 6:
        return 0.1549
    return 0.4646 if h >= 17 else 0.1807


C3 = np.array([tariff3(h, o) for h, o in zip(hour, off)])
C1 = np.full(n, 0.2581)
inflex = df["inflex"].values[t0:]

J = jobs[(jobs.release >= t0) & (jobs.deadline <= t0 + n)].copy()
J["release"] -= t0
J["deadline"] -= t0
J = J.sort_values(["release", "deadline"]).reset_index(drop=True)
# задания, пересекающие границы периода, выполняются без управления во всех сценариях
outside = jobs[~jobs.index.isin(jobs[(jobs.release >= t0) & (jobs.deadline <= t0 + n)].index)]
fixed = np.zeros(n)
unc_all = df[["ev_unc", "boiler_unc", "wash_unc", "dish_unc"]].sum(axis=1).values[t0:]


def run_uncontrolled(j, load):
    if j.profile:
        for k, q in enumerate(map(float, j.profile.split(";"))):
            load[j.release + k] += q
    else:
        t, e = j.release, j.energy
        while e > 1e-9:
            q = min(j.pmax, e); load[t] += q; e -= q; t += 1


# фон: гибкая нагрузка заданий вне периода (одинакова во всех сценариях)
inside_unc = np.zeros(n)
for _, j in J.iterrows():
    run_uncontrolled(j, inside_unc)
fixed = unc_all - inside_unc
fixed[fixed < 1e-9] = 0


def schedule_uncontrolled():
    load = np.zeros(n)
    for _, j in J.iterrows():
        run_uncontrolled(j, load)
    return load


def schedule_timer():
    """Простой таймер: в будни всё, что появилось в пиковой зоне 17–23 ч, стартует в 23:00 на полной мощности."""
    load = np.zeros(n)
    for _, j in J.iterrows():
        h = hour[j.release]
        if not off[j.release] and 17 <= h <= 22:
            j = j.copy(); j.release = j.release + (23 - h)
        run_uncontrolled(j, load)
    return load


def dcost(L, t, d, c):
    """Приращение целевой функции при добавлении d кВт·ч в час t."""
    new = L[t] + d
    v = c[t] * d + MU * (new ** 2 - L[t] ** 2)
    if t > 0:
        v += NU * ((new - L[t - 1]) ** 2 - (L[t] - L[t - 1]) ** 2)
    if t < n - 1:
        v += NU * ((L[t + 1] - new) ** 2 - (L[t + 1] - L[t]) ** 2)
    v += 5.0 * (max(new - P_LIM, 0) - max(L[t] - P_LIM, 0))
    return v


def schedule_smart(base_forecast, c=C3):
    L = base_forecast + fixed            # плановая нагрузка
    flex = np.zeros(n)
    for _, j in J.iterrows():
        r, dl = int(j.release), int(j.deadline)
        if j.profile:                    # неделимая программа: перебор старта
            prof = list(map(float, j.profile.split(";")))
            best, best_s = np.inf, r
            for s in range(r, dl - len(prof) + 1):
                Lt = L.copy(); tot = 0
                for k, q in enumerate(prof):
                    tot += dcost(Lt, s + k, q, c); Lt[s + k] += q
                if tot < best - 1e-12:
                    best, best_s = tot, s
            for k, q in enumerate(prof):
                L[best_s + k] += q; flex[best_s + k] += q
        else:                            # делимая нагрузка: заполнение провала квантами
            alloc = np.zeros(n); e = j.energy
            while e > 1e-9:
                d = min(Q, e)
                cand = [t for t in range(r, dl) if alloc[t] + d <= j.pmax + 1e-9]
                t = min(cand, key=lambda t: dcost(L, t, d, c))
                L[t] += d; alloc[t] += d; flex[t] += d; e -= d
    return flex


def metrics(flex, name, c=C3):
    load = inflex + fixed + flex
    s = pd.Series(load, index=idx)
    day = s.groupby(idx.normalize())
    wd = ~off
    peak_zone = wd & (hour >= 17) & (hour <= 22)
    h22, h23 = (hour == 22) & wd, (hour == 23) & wd
    return {
        "Сценарий": name,
        "Энергия, кВт·ч": load.sum(),
        "Макс. нагрузка, кВт": load.max(),
        "Средний суточный максимум, кВт": day.max().mean(),
        "Коэффициент неравномерности": (day.max() / day.mean()).mean(),
        "Доля энергии в пиковой зоне (будни), %": load[peak_zone].sum() / load[wd].sum() * 100,
        "Средняя нагрузка 23–24 ч (будни), кВт": load[h23].mean(),
        "Средний скачок 22→23 ч, кВт": (load[h23] - load[h22]).mean(),
        "Часы сверх 7 кВт": int((load > P_LIM + 1e-9).sum()),
        "Затраты, трёхзонный тариф, BYN": (load * c).sum(),
        "Затраты, одноставочный тариф, BYN": (load * C1).sum(),
    }


if __name__ == "__main__":
    S = {}
    S["Без управления"] = schedule_uncontrolled()
    S["Таймер на 23:00"] = schedule_timer()
    S["Алгоритм (прогноз: случайный лес)"] = schedule_smart(fc["Случайный лес"].values)
    S["Алгоритм (прогноз: градиентный бустинг)"] = schedule_smart(fc["Градиентный бустинг"].values)
    S["Алгоритм (прогноз: линейная регрессия)"] = schedule_smart(fc["Линейная регрессия"].values)
    S["Алгоритм (наивный прогноз)"] = schedule_smart(fc["Наивный (неделю назад)"].values)
    S["Алгоритм (идеальный прогноз)"] = schedule_smart(inflex)
    rows = [metrics(v, k) for k, v in S.items()]
    res = pd.DataFrame(rows).set_index("Сценарий")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 20)
    print(res.round(2).T)
    res.to_csv("control_results.csv", float_format="%.3f")
    prof = pd.DataFrame({k: inflex + fixed + v for k, v in S.items()}, index=idx)
    prof["inflex"] = inflex
    prof["off_day"] = off.astype(int)
    prof.to_csv("control_profiles.csv", float_format="%.4f")
