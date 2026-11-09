"""米国10年金利の20営業日変化による『金利低下／上昇』シグナルが、1年のうちどれだけの期間出ているか"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
x = pd.read_pickle(SP + "crisis_ext.pkl")["^TNX"].dropna()
x = x[x.index >= "2007-01-01"]
ch = x.diff(20)
wk = ch.resample("W-FRI").last().dropna()          # 週末（シグナルを判定する時点）
wk = wk[wk.index >= "2008-01-01"]
print("データ期間", x.index.min().date(), "〜", x.index.max().date(), " 最新の20日変化", round(ch.iloc[-1], 3), "（", ch.index[-1].date(), "、10年金利", round(x.iloc[-1], 3), "%）")
defs = {"RSI2戦術の低下(≤-0.13)": wk <= -0.13, "RSI2戦術の上昇(>+0.06)": wk > 0.06,
        "RSI14戦術の大きな低下(≤-0.27)": wk <= -0.27, "RSI14戦術の上昇(>-0.03)": wk > -0.03}
T = pd.DataFrame({k: v.groupby(v.index.year).mean() * 100 for k, v in defs.items()})
T["週数"] = wk.groupby(wk.index.year).size(); T["10年金利の年間変化"] = x.resample("YE").last().diff().reindex(pd.to_datetime([f"{y}-12-31" for y in T.index]), method="nearest").values
T.loc["全期間平均"] = [v.mean() * 100 for v in defs.values()] + [len(wk), np.nan]
pd.set_option("display.width", 200); print("\n【年ごと：週末の判定でシグナルが出ていた週の割合 %】"); print(T.round(1).to_string())
for lab, a, b in [("2008-2017", "2008", "2017"), ("2018-2022", "2018", "2022"), ("2023-2026", "2023", "2026")]:
    print(f"  {lab}: " + " ／ ".join(f"{k} {v.loc[a:b].mean()*100:.0f}%" for k, v in defs.items()))
print("\n【連続する期間の長さ（週）】")
for k, v in defs.items():
    runs = (v != v.shift()).cumsum(); L = v.groupby(runs).agg(["first", "size"]); L = L[L["first"]]["size"]
    print(f"  {k}: 回数{len(L)}（年{len(L)/len(set(v.index.year)):.1f}回） 平均{L.mean():.1f}週 中央{L.median():.0f}週 最長{L.max()}週 1週だけ{(L==1).mean()*100:.0f}%")
print("\n【月ごと（全期間）：シグナルが出ていた週の割合 %】")
M = pd.DataFrame({k: v.groupby(v.index.month).mean() * 100 for k, v in defs.items()}); print(M.round(0).T.to_string())
print("\n【20日変化の分布（週末、全期間）】", wk.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_dict())
print("\n【直近12週の判定】"); print(pd.DataFrame({"20日変化": wk.tail(12).round(3), **{k: v.tail(12).map({True: "○", False: "・"}) for k, v in defs.items()}}).to_string())
