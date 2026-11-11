"""
Публикация 2. Модель дома сценария C (публикация 1): почасовое электропотребление
за 2025 г. с реальной температурой воздуха в Минске (NASA POWER, UTC -> UTC+3).

Нагрузка делится на:
  * негибкую (базовая нагрузка, кухня, кондиционер) — её прогнозирует ML-модель;
  * гибкую — задания (jobs) с моментом появления (release), крайним сроком (deadline),
    допустимой мощностью и профилем: зарядка электромобиля, бойлер, стирка+сушка,
    посудомоечная машина. В сценарии «без управления» каждое задание выполняется
    сразу после появления.
Результат: house_c_hourly.csv (почасовые ряды) и house_c_jobs.csv (задания).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")  # кириллица и символы в консоли Windows
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(2025)

# ---------------------------------------------------------------- календарь и погода
idx = pd.date_range("2025-01-01 00:00", "2025-12-31 23:00", freq="h")
N = len(idx)
w = json.load(open("minsk_weather_2025_nasa.json"))["properties"]["parameter"]["T2M"]
t_utc = pd.Series({pd.Timestamp(k[:8] + " " + k[8:] + ":00"): v for k, v in w.items()}).sort_index()
t_local = t_utc.copy()
t_local.index = t_local.index + pd.Timedelta(hours=3)  # Беларусь: UTC+3 круглый год
T = t_local.reindex(idx).interpolate().bfill().values

HOLIDAYS = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-07", "2025-03-08", "2025-04-29",
                           "2025-05-01", "2025-05-09", "2025-07-03", "2025-11-07", "2025-12-25"])
days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
off_day = {d: (d.dayofweek >= 5) or (d in HOLIDAYS) for d in days}
t_daymean = pd.Series(T, index=idx).resample("D").mean()

# долгота светового дня для широты Минска и восход/закат по местному времени
doy = idx.dayofyear.values
daylen = 12 + 4.9 * np.sin(2 * np.pi * (doy - 80) / 365)
noon = 13.15  # солнечный полдень в Минске по UTC+3
hour = idx.hour.values
dark = (hour + 0.5 < noon - daylen / 2) | (hour + 0.5 > noon + daylen / 2)

H = {d: np.where((idx.normalize() == d))[0][0] for d in days}  # индекс 0 ч каждого дня

base = np.zeros(N); kitchen = np.zeros(N); ac = np.zeros(N)
jobs = []  # dict(type, release, deadline, energy, pmax, profile|None, uncontrolled start)


def add_block(arr, start, energy, pmax):
    """Непрерывная работа с мощностью pmax до набора energy (кВт·ч) — без управления."""
    t, e = start, energy
    while e > 1e-9 and t < N:
        q = min(pmax, e)
        arr[t] += q
        e -= q
        t += 1


# ---------------------------------------------------------------- присутствие жильцов
home = np.zeros(N, dtype=bool)
awake = np.zeros(N, dtype=bool)
wake_h, leave_h, back_h, sleep_h = {}, {}, {}, {}
for d in days:
    h0 = H[d]
    off = off_day[d]
    wfh = (not off) and rng.random() < 0.12
    wk = (6 if rng.random() < 0.85 else 7) if not off else int(rng.choice([8, 9, 9, 10]))
    sl = int(rng.choice([22, 23, 23, 24]))
    if off or wfh:
        lv, bk = (int(rng.integers(11, 14)), int(rng.integers(15, 19))) if (off and rng.random() < 0.5) else (None, None)
    else:
        lv = 8
        typical = {0: 18, 1: 18, 2: 19, 3: 18, 4: 17}[d.dayofweek]   # привычный график по дням недели
        bk = typical if rng.random() < 0.75 else int(rng.choice([17, 18, 19, 20]))
    wake_h[d], leave_h[d], back_h[d], sleep_h[d] = wk, lv, bk, sl
    for h in range(24):
        aw = wk <= h < sl
        at_home = not (lv is not None and lv <= h < bk)
        awake[h0 + h] = aw and at_home
        home[h0 + h] = at_home

# ---------------------------------------------------------------- негибкая нагрузка
for i in range(N):
    p = 0.045 + 0.0008 * max(T[i], 0)                       # холодильник, морозильник
    p += 0.022                                              # роутер, видеонаблюдение, дежурные режимы
    if T[i] < 10:
        p += 0.035                                          # насос газового котла
    if awake[i]:
        p += 0.07 + (0.08 if hour[i] >= 19 else 0.0)        # электроника, ТВ, ПК
        if dark[i]:
            p += 0.10 + (0.05 if hour[i] >= 19 else 0.0)    # освещение
    if dark[i] and (17 <= hour[i] <= 23 or 5 <= hour[i] <= 7):
        p += 0.03                                           # наружное освещение
    base[i] = p * rng.lognormal(0, 0.12)
# случайные бытовые приборы (утюг, пылесос, фен, инструмент)
for d in days:
    for _ in range(rng.poisson(0.8 if off_day[d] else 0.4)):
        cand = [H[d] + h for h in range(24) if awake[H[d] + h]]
        if cand:
            base[rng.choice(cand)] += rng.uniform(0.2, 0.8)

for d in days:
    h0, off = H[d], off_day[d]
    # завтрак
    bh = wake_h[d] + (0 if not off else int(rng.integers(0, 2)))
    if awake[h0 + bh]:
        kitchen[h0 + bh] += rng.uniform(0.3, 0.7) * (1.6 if off else 1.0)
    # обед в выходной
    if off and awake[h0 + 13]:
        add_block(kitchen, h0 + int(rng.choice([12, 13, 14])), rng.uniform(0.4, 1.4), 1.6)
    # ужин: через 0–1 ч после возвращения
    start = (back_h[d] if back_h[d] is not None else 18) + (0 if rng.random() < 0.8 else 1)
    e_dinner = rng.uniform(0.8, 2.2) + (rng.uniform(0.8, 1.6) if rng.random() < 0.22 else 0)  # +духовка
    add_block(kitchen, h0 + start, e_dinner, rng.uniform(1.4, 2.6))
    # чайник, СВЧ вечером
    if awake[h0 + 21]:
        kitchen[h0 + 21] += rng.uniform(0.0, 0.25)

# кондиционер
for i in range(N):
    if T[i] > 21.0 and home[i] and 12 <= hour[i] <= 23:
        ac[i] = min(1.1, 0.35 + 0.15 * (T[i] - 21.0)) * rng.uniform(0.8, 1.1)

# ---------------------------------------------------------------- гибкие нагрузки
def next_day(d):
    return d + pd.Timedelta(days=1)


ev_deficit = rng.uniform(0, 8)
for d in days:
    h0, off = H[d], off_day[d]
    dn = next_day(d)
    dep_next = 7 if (dn in off_day and not off_day[dn]) else 9    # к этому часу всё должно быть готово
    deadline_morning = H[d] + 24 + dep_next if dn in off_day else N
    deadline_morning = min(deadline_morning, N)

    # --- бойлер: утренний и вечерний водоразбор, догрев 2 кВт
    e_dhw = 5.5 * (1 + 0.16 * np.cos(2 * np.pi * (d.dayofyear - 20) / 365)) * rng.uniform(0.8, 1.2)
    m_h = h0 + wake_h[d] + 1
    e_m = 0.4 * e_dhw
    jobs.append(dict(type="boiler", release=m_h, deadline=h0 + 17, energy=e_m, pmax=2.0, profile=None))
    ev_h = h0 + min(sleep_h[d] - 1, 22) if sleep_h[d] <= 23 else h0 + 22
    jobs.append(dict(type="boiler", release=ev_h, deadline=min(h0 + 24 + 6, N), energy=e_dhw - e_m,
                     pmax=2.0, profile=None))

    # --- электромобиль: суточный пробег, расход зависит от температуры
    km = rng.normal(30, 8) if not off else (rng.choice([0, 20, 60, 160], p=[0.3, 0.35, 0.25, 0.1]))
    km = max(km, 0) + (rng.uniform(5, 25) if rng.random() < 0.4 else 0)
    cons = 0.17 + 0.0055 * max(15 - t_daymean[d], 0)          # кВт·ч/км с учётом потерь зарядки
    ev_deficit += 0.85 * km * cons                              # ~15 % энергии заряжается вне дома
    arr = back_h[d] if back_h[d] is not None else int(rng.integers(15, 20))
    if ev_deficit > 10 and arr is not None:
        jobs.append(dict(type="ev", release=h0 + arr, deadline=deadline_morning, energy=ev_deficit,
                         pmax=7.4, profile=None))
        ev_deficit = 0.0

    # --- стирка (+ сушка) и посудомойка: неделимые программы
    if rng.random() < 4 / 7:
        st = int(rng.choice([19, 20, 21])) if not off else int(rng.integers(10, 18))
        prof = [0.75, 0.15] + ([0.7, 0.6] if rng.random() < 0.75 else [])
        jobs.append(dict(type="wash", release=h0 + st, deadline=deadline_morning, energy=sum(prof),
                         pmax=None, profile=prof))
    if rng.random() < 5 / 7:
        st = int(rng.choice([20, 21, 21, 22])) if not (off and rng.random() < 0.4) else int(rng.integers(14, 17))
        prof = [0.6, 0.35]
        jobs.append(dict(type="dish", release=h0 + st, deadline=deadline_morning, energy=sum(prof),
                         pmax=None, profile=prof))

jobs = pd.DataFrame(jobs)
jobs = jobs[jobs.release < N - 30].reset_index(drop=True)
jobs["profile"] = jobs["profile"].apply(lambda p: "" if p is None else ";".join(f"{x:.3f}" for x in p))

# ---------------------------------------------------------------- сценарий «без управления»
flex_unc = {k: np.zeros(N) for k in ["ev", "boiler", "wash", "dish"]}
for _, j in jobs.iterrows():
    if j.profile:
        for k, q in enumerate(map(float, j.profile.split(";"))):
            flex_unc[j.type][j.release + k] += q
    else:
        add_block(flex_unc[j.type], j.release, j.energy, j.pmax)

df = pd.DataFrame(index=idx)
df.index.name = "timestamp"
df["hour"], df["dow"] = hour, idx.dayofweek
df["off_day"] = [int(off_day[d]) for d in idx.normalize()]
df["temp"] = T.round(2)
df["dark"] = dark.astype(int)
df["base"], df["kitchen"], df["ac"] = base, kitchen, ac
df["inflex"] = base + kitchen + ac
for k, v in flex_unc.items():
    df[f"{k}_unc"] = v
df["total_unc"] = df["inflex"] + sum(flex_unc.values())
df.round(4).to_csv("house_c_hourly.csv")
jobs.to_csv("house_c_jobs.csv", index=False)

year = df.sum()
print("Год, кВт·ч:", {k: round(year[k]) for k in ["base", "kitchen", "ac", "inflex", "ev_unc",
                                                 "boiler_unc", "wash_unc", "dish_unc", "total_unc"]})
daily = df["total_unc"].resample("D")
print(f"Среднесуточное: {daily.sum().mean():.1f} кВт·ч; средний суточный максимум: {daily.max().mean():.2f} кВт; "
      f"максимум: {df.total_unc.max():.2f} кВт")
ev_days = df["ev_unc"].resample("D").sum() > 0
print(f"Дни с зарядкой ЭМ: {ev_days.sum()}; в них суточное {daily.sum()[ev_days].mean():.1f} кВт·ч, "
      f"максимум {daily.max()[ev_days].mean():.2f} кВт")
print("Заданий:", jobs.type.value_counts().to_dict())
