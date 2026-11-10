"""screen_rsi14.py の結果に株探の決算日を当て、必須条件を判定"""
import sys, json, pandas as pd
d = pd.Timestamp(sys.argv[1]); K = json.load(open(sys.argv[2]))
R = pd.read_csv(f"screen_rsi14_{d.date()}.csv", dtype={"コード": str}, parse_dates=["Yahoo次の決算", "Yahoo直近4週"])
def lk(c):
    L = sorted(K.get(c, []), key=lambda r: r[1]); return (pd.Timestamp(L[-1][1][1:9]), L[-1][2][:38]) if L else (pd.NaT, "")
R["株探直近"] = [lk(c)[0] for c in R.コード]; R["見出し"] = [lk(c)[1] for c in R.コード]
R["前4週"] = (R.株探直近 > d - pd.Timedelta(days=28)) | R.Yahoo直近4週.notna()
R["後2週"] = (R.Yahoo次の決算 <= d + pd.Timedelta(days=14)) | (R.株探直近.between("2026-07-15", "2026-09-30"))
R["配当OK"] = R["配当利回り%"] >= 2.45
R["合格"] = ~R.前4週 & ~R.後2週 & R.業界ぐるみ & R.配当OK
pd.set_option("display.width", 320)
print(f"{d.date()}  RSI14<30 {len(R)}  決算の前後でない {int((~R.前4週 & ~R.後2週).sum())}  ＋業界ぐるみ {int((~R.前4週 & ~R.後2週 & R.業界ぐるみ).sum())}  ＋配当2.45%以上 {int(R.合格.sum())}")
cols = ["コード", "銘柄", "業種", "終値", "RSI14", "25日乖離%", "業種内25日乖離割合%", "業種比20日%", "業界ぐるみ", "配当利回り%", "株探直近", "見出し", "Yahoo次の決算", "前4週", "後2週"]
print(R[~R.前4週 & ~R.後2週][cols].round(1).to_string(index=False))
R.to_csv(f"screen_rsi14_{d.date()}_judged.csv", index=False)
