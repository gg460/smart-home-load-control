"""Рисунки публикации 2 (JPEG): текст не мельче 12 pt, линии не тоньше 1 pt при ширине 16,5 см."""
import sys
sys.stdout.reconfigure(encoding="utf-8")  # кириллица и символы в консоли Windows
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.edgecolor": "#52514e", "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#d9d8d2", "grid.linewidth": 1.0,
    "xtick.color": "#0b0b0b", "ytick.color": "#0b0b0b", "savefig.dpi": 300,
})
for side in ("top", "right"):
    plt.rcParams[f"axes.spines.{side}"] = False
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2 = "#0b0b0b", "#52514e"
CM = 1 / 2.54
W = 16.5 * CM


def save(fig, name):
    fig.savefig(name, format="jpg", pil_kwargs={"quality": 95}, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ------------------------------------------------------------- рис. 1: схема системы
fig, ax = plt.subplots(figsize=(W, 7.0 * CM))
ax.set_xlim(0, 100); ax.set_ylim(0, 42.5); ax.axis("off")


def box(x, y, w, h, text, fc="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec=INK, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, linespacing=1.1)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK, mutation_scale=11))


BH = 12.5
L_ = [(29, "Умный счётчик,\nдатчики тока"), (15, "Тариф, прогноз\nпогоды"), (1, "Сроки\nготовности\n(комфорт)")]
R_ = [(29, "Зарядная станция\nэлектромобиля\n(ток 6–32 А)"), (15, "Реле бойлера"),
      (1, "Стиральная,\nсушильная,\nпосудомоечная машины")]
for y, t in L_:
    box(0.5, y, 26, BH, t)
for y, t in R_:
    box(64, y, 35.5, BH, t)
box(34, 6, 23, 30, "Локальный\nконтроллер:\nпрогноз (ML)\nи планиров-\nщик гибких\nнагрузок", fc="#eaf2fc")
for y, _ in L_:
    arrow(27.2, y + BH / 2, 33.4, 21.25 + (y - 15) * 0.55)
for y, _ in R_:
    arrow(57.6, 21.25 + (y - 15) * 0.55, 63.4, y + BH / 2)
save(fig, "fig1_system.jpg")

# ------------------------------------------------------------- рис. 2: прогноз и факт
f = pd.read_csv("forecast_test.csv", parse_dates=["timestamp"], index_col="timestamp")
wk = f.loc["2025-12-08":"2025-12-11"]
fig, ax = plt.subplots(figsize=(W, 6.0 * CM))
x = np.arange(len(wk))
ax.plot(x, wk["fact"], color=INK, lw=1.4, label="Факт")
ax.plot(x, wk["Случайный лес"], color=C2, lw=1.6, label="Случайный лес")
ax.plot(x, wk["Линейная регрессия"], color=C1, lw=1.4, ls="--", label="Линейная регрессия")
ticks = list(range(0, len(wk), 12))
ax.set_xticks(ticks)
ax.set_xticklabels([("{:%d.%m}\n".format(wk.index[i]) if wk.index[i].hour == 0 else "\n") + f"{wk.index[i].hour} ч"
                    for i in ticks])
ax.set_xlim(0, len(wk) - 1)
ax.set_ylabel("Мощность, кВт")
ax.set_ylim(0, 3.2)
ax.legend(loc="upper left", frameon=False, ncol=3, handlelength=1.8)
ax.grid(axis="x", visible=False)
save(fig, "fig2_forecast.jpg")

# ------------------------------------------------------------- рис. 3: профиль до/после
p = pd.read_csv("control_profiles.csv", parse_dates=["timestamp"], index_col="timestamp")
h = pd.read_csv("house_c_hourly.csv", parse_dates=["timestamp"], index_col="timestamp")
j = pd.read_csv("house_c_jobs.csv")
cols = {"Без управления": (C2, "-", "Без управления"),
        "Таймер на 23:00": (INK2, "--", "Таймер на 23:00"),
        "Алгоритм (прогноз: случайный лес)": (C1, "-", "Предлагаемый алгоритм")}
ev_days = set(h.index[j[j.type == "ev"].release].normalize())
wins = []
for d in pd.date_range("2025-11-01", "2025-12-30"):
    if d in ev_days and h.loc[d.strftime("%Y-%m-%d"), "off_day"].iloc[0] == 0:
        w = p.loc[d + pd.Timedelta(hours=12): d + pd.Timedelta(hours=35), list(cols)]
        if len(w) == 24:
            wins.append(w.values)
avg = np.mean(wins, axis=0)
print("Окон:", len(wins), "энергия:", avg.sum(0).round(1), "максимум:", avg.max(0).round(2),
      "Kнер:", (avg.max(0) / avg.mean(0)).round(2))
hrs = list(range(12, 24)) + list(range(0, 12))
xs = np.arange(25)
fig, ax = plt.subplots(figsize=(W, 6.4 * CM))
ax.axvspan(5, 11, color="#fde9c2", lw=0, zorder=0)
ax.axvspan(11, 18, color="#dcebfb", lw=0, zorder=0)
ax.text(8, 10.35, "пик 17–23 ч", ha="center", va="top", fontsize=12, color=INK2)
ax.text(14.5, 10.35, "ночь 23–6 ч", ha="center", va="top", fontsize=12, color=INK2)
for k, (col, (c, ls, lab)) in enumerate(cols.items()):
    yv = np.r_[avg[:, k], avg[-1, k]]
    ax.step(xs, yv, where="post", color=c, ls=ls, lw=1.8, label=lab)
ax.set_xticks(range(0, 25, 3))
ax.set_xticklabels([f"{(12 + t) % 24}" for t in range(0, 25, 3)])
ax.set_xlim(0, 24)
ax.set_ylim(0, 10.5)
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.set_xlabel("Час суток")
ax.set_ylabel("Мощность, кВт")
ax.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.45, -0.22), ncol=3, handlelength=1.6, columnspacing=1.0)
ax.grid(axis="x", visible=False)
save(fig, "fig3_profile.jpg")
print("Готово.")
