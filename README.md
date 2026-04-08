# BASS x NBD-Dirichlet 融合モデル

Julia + Turing.jl によるベイズ推定で、新商品のプロモーション効果を **試用（Trial）** と **リピート（Repeat）** に分離して判定する。

## モデル概要

### 1. 拡張 BASS モデル（試用拡散）

新商品の採用プロセスを Bass 拡散モデルで記述し、プロモーション効果 γp を推定する。

```
λ_trial(t) = (p · exp(γp · Promo(t)) + q · N(t-1)/M) · (M - N(t-1))
```

### 2. 集約 NBD（リピート購買）

既存購買者のリピート頻度を負の二項分布でモデル化し、プロモーション効果 δr を推定する。

```
n_repeat[t] ~ NegBin(N(t-1)·K_r, K_r/(K_r + μ_r · exp(δr · Promo(t))))
```

### 3. NBD-Dirichlet（ブランド選択）

カテゴリ内のブランド選択構造を Dirichlet-Multinomial でモデル化する。

```
n_cat[i] ~ NegBin(K_d, K_d/(K_d + M_d))
brand_counts[i,:] ~ DirichletMultinomial(n_cat[i], S_d · p_d)
```

## データソース

| データ | 説明 |
|--------|------|
| [dunnhumby "The Complete Journey"](https://www.dunnhumby.com/source-files/) | 約 2,500 世帯の 2 年間（102 週）の小売購買データ |
| [Craft Beer Bar (Kaggle)](https://www.kaggle.com/datasets/podsyp/sales-in-craft-beer-bar) | クラフトビールバーの販売データ（Gaffel ブランド分析用） |
| 100 本ノック POS データ | 飲料カテゴリの週次購買データ |

## ノートブック

| ファイル | 内容 |
|---------|------|
| `BASS_NBD_Dirichlet_Promo.ipynb` | 3 層融合モデル（BASS + NBD + Dirichlet） |
| `beer_NBD-Dirichlet_analysis.ipynb` | dunnhumby Beer (Week 20-30) の NBD-Dirichlet 分析 |
| `beer_data.ipynb` | Beer 購買データの探索・可視化 |
| `dunnhumby_db_summary.ipynb` | dunnhumby.db の構造確認 |

## 分析結果（Beer Week 20-30）

NBD-Dirichlet モデルによる Beer カテゴリ（54 ブランド）の推定:

- **M** — カテゴリ平均購買回数
- **K** — NBD 集約パラメータ（購買頻度の異質性）
- **S** — Dirichlet 集約パラメータ（ブランド選択の異質性）

Double Jeopardy 法則（小シェアブランドほど浸透率・購買頻度が低い）の検証を含む。

## 技術スタック

- **Julia 1.11** + Jupyter
- **Turing.jl** — ベイズ MCMC 推定（NUTS サンプラー）
- **ModelingToolkit.jl** — ODE 定義（BASS モデル）
- **SQLite.jl** — dunnhumby.db アクセス
- **StatsPlots.jl** — 可視化

## セットアップ

```julia
using Pkg
Pkg.add([
    "Turing", "Distributions", "StatsPlots", "DataFrames",
    "SpecialFunctions", "MCMCChains", "StatsBase",
    "ModelingToolkit", "DifferentialEquations",
    "SQLite", "CSV"
])
```

## dunnhumby.db

`dunnhumby.db`（SQLite）は容量が大きいためリポジトリには含まれません。
[dunnhumby](https://www.dunnhumby.com/source-files/) から CSV をダウンロードし、
`SqlNavigatorCli` の `import-dir` コマンドでインポートしてください。

```bash
# DB に含まれるテーブル
campaign_desc, campaign_table, causal_data, coupon, coupon_redempt,
hh_demographic, product, transaction_data,
beer_20_30_transactions, beer_20_30_weekly_brand, beer_20_30_weekly_total
```

## ファイル構成

```
.
├── BASS_NBD_Dirichlet_Promo.ipynb   # BASS × NBD-Dirichlet 融合モデル
├── beer_NBD-Dirichlet_analysis.ipynb # Beer NBD-Dirichlet 分析 (Week 20-30)
├── beer_data.ipynb                   # Beer データ探索
├── dunnhumby_db_summary.ipynb        # DB 構造確認
├── create_notebook.py                # ノートブック生成スクリプト
├── test_notebook.jl                  # モデル検証スクリプト
├── CLAUDE.md                         # Claude Code 向け指示
├── 100knock/                         # POS データ 100 本ノック
└── beer_data/                        # Kaggle Beer データ
```

## 参考文献

- Bass, F. M. (1969). "A new product growth for model consumer durables." *Management Science*, 15(5).
- Goodhardt, G. J., Ehrenberg, A. S. C., & Chatfield, C. (1984). "The Dirichlet: A comprehensive model of buying behaviour." *JRSS Series A*, 147(5).
- Ehrenberg, A. S. C. (1988). *Repeat-Buying: Facts, Theory and Applications.*
