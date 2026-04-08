# CLAUDE.md — BASS × NBD-Dirichlet 融合モデル

## プロジェクト概要

Julia + ModelingToolkit.jl + Turing.jl による BASS × NBD-Dirichlet 融合モデル。
新商品（飲料カテゴリ）のプロモーション効果を試用（Trial）とリピート（Repeat）に分離して判定する。

## ファイル構成

| ファイル | 内容 |
|---------|------|
| `BASS_NBD_Dirichlet_Promo.ipynb` | メインノートブック。BASS拡張 + 集約NBD + Dirichletの3層モデル |
| `beverage_weekly.csv` | 前処理済み週次データ: week, n_trial, n_repeat, promo, cum_trial |
| `beverage_brand_matrix.csv` | 消費者×ブランド購買回数行列（Dirichlet用） |
| `create_notebook.py` | ノートブック生成スクリプト |
| `100knock/` | 元データ（POSデータ 100本ノック） |

## データ仕様

### beverage_weekly.csv
- 飲料カテゴリ(0710) の 148 週間データ（2017-01-01 ~ 2019-10-31）
- `n_trial`: その週の新規飲料購買者数（初回購買）
- `n_repeat`: その週の既存購買者のリピート購買回数
- `promo`: P75 超の週 = 1（プロモ活動期間のプロキシ）
- 3,325 ユニーク消費者、7 ブランド（サブカテゴリ）

### beverage_brand_matrix.csv
- 3,325 消費者 × 7 ブランドの購買回数行列
- ブランド = サブカテゴリ: 果汁・清涼飲料 / お茶飲料 / コーヒー飲料 / 機能性飲料 / 炭酸飲料 / 水 / その他

## モデル構造

### 1. 拡張BASS（試用拡散）
```
λ_trial(t) = (p · exp(γp · Promo(t)) + q · N(t-1)/M) · (M - N(t-1))
n_trial[t] ~ Normal(λ, σ + √λ)
```
- γp > 0: プロモーションが試用を加速
- ModelingToolkit.jl で ODE 定義、Turing.jl で離散差分ベイズ推定

### 2. 集約NBD（リピート購買）
```
n_repeat[t] ~ NegBin(N(t-1)·K_r, K_r/(K_r + μ_r · exp(δr · Promo(t))))
```
- δr > 0: プロモーションがリピート率を押し上げ

### 3. NBD-Dirichlet（ブランド選択）
```
n_cat[i] ~ NegBin(K_d, K_d/(K_d+M_d))
brand_counts[i,:] ~ DirichletMultinomial(n_cat[i], S_d · p_d)
```

## 判定基準
- 95% CI 下限 > 0 → 有意な促進効果あり
- P(param > 0 | data) > 0.90 → 効果の傾向あり
- exp(γp), exp(δr) → 効果の倍率

## Julia / パッケージ上の注意

### ModelingToolkit.jl
- `@register_symbolic` でプロモ指標関数を ODE に埋め込む
- `mtkcompile()` で ODESystem をコンパイル（MTK v9+）

### Turing.jl + ForwardDiff
- 累積変数: `N_cum = zero(p_v)` で型を合わせる
- ベクトル: `Vector{typeof(p_v)}(undef, T)` で Dual 型対応
- 非負保護: `raw > 0 ? raw : zero(raw)` でReLU（max(x, Float64)は型不一致の恐れ）
- NegativeBinomial: `r_nb = N_prev * K_r` の集約NBDパラメータは非整数 OK（Distributions.jl）
- **NegativeBinomial + ForwardDiff の DomainError**: NUTS探索中にDual数のパラメータがコンストラクタの引数チェック(`0 < p <= 1`)で例外を起こす。**必ず `check_args=false` を付ける**:
  ```julia
  n_repeat[t] ~ NegativeBinomial(r_nb, p_nb; check_args=false)
  ```

### MCMCChains API
- `vec(chain[:param])` でサンプル取得
- `get(chain, :param).param` でスカラー、`namesingroup(chain, :p_d)` でベクトル
- `Array(chain)` は使わない（バージョン依存で不安定）

### CSV.jl
- BOM 付きCSFの場合: `CSV.read(path, DataFrame)` で自動処理される

## 使用パッケージ一覧

```julia
using ModelingToolkit, DifferentialEquations, Turing
using Distributions, StatsPlots, DataFrames, MCMCChains
using SpecialFunctions, CSV, Random, Statistics, Printf
```
