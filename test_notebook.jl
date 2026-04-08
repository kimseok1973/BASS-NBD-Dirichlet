#!/usr/bin/env julia
# Test script: runs notebook cells sequentially and reports errors
println("=" ^ 60)
println("BASS × NBD-Dirichlet Notebook Verification")
println("=" ^ 60)

cd(@__DIR__)

# ============================================================
# Cell 2: Using
# ============================================================
println("\n[Cell 2] Loading packages...")
@time begin
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D
    using DifferentialEquations
    using Turing
    using Distributions
    using StatsPlots
    using DataFrames
    using MCMCChains
    using SpecialFunctions
    using CSV
    using Random
    using Statistics
    using Printf
end
Random.seed!(2024)
println("  Julia version: ", VERSION)
println("  ✓ All packages loaded")

# ============================================================
# Cell 3: Font (skip in headless mode, just set env)
# ============================================================
println("\n[Cell 3] Plot backend setup...")
ENV["GKS_ENCODING"] = "utf8"
gr()
println("  ✓ GR backend set")

# ============================================================
# Cell 5: Data load
# ============================================================
println("\n[Cell 5] Loading data...")
df_weekly = CSV.read("beverage_weekly.csv", DataFrame)
df_brand  = CSV.read("beverage_brand_matrix.csv", DataFrame)

println("  週次データ: $(nrow(df_weekly)) 週")
println("  ブランドデータ: $(nrow(df_brand)) 消費者")
println("  ✓ Data loaded")

# ============================================================
# Cell 6: Data prep
# ============================================================
println("\n[Cell 6] Data preparation...")
T_PERIODS    = nrow(df_weekly)
weeks        = df_weekly.week
n_trial_obs  = df_weekly.n_trial
n_repeat_obs = df_weekly.n_repeat
promo_vec    = df_weekly.promo
N_cum_obs    = df_weekly.cum_trial

println("  T=$T_PERIODS, cum_trial=$(N_cum_obs[end]), promo_weeks=$(sum(promo_vec))")
println("  ✓ Variables ready")

# ============================================================
# Cell 8: MTK ODE definition
# ============================================================
println("\n[Cell 8] MTK ODE definition...")
const PROMO_DATA = Float64.(promo_vec)

function promo_signal(t_val::Real)
    idx = clamp(round(Int, t_val), 1, length(PROMO_DATA))
    return PROMO_DATA[idx]
end
@register_symbolic promo_signal(t)

@parameters p_bass q_bass M_bass gp_param
@variables N_bass(t) n_bass(t)

eqs_bass_promo = [
    D(N_bass) ~ (p_bass * exp(gp_param * promo_signal(t)) + q_bass * N_bass / M_bass) *
                 (M_bass - N_bass),
    n_bass    ~ (p_bass * exp(gp_param * promo_signal(t)) + q_bass * N_bass / M_bass) *
                 (M_bass - N_bass)
]

@named bass_promo_sys = ODESystem(eqs_bass_promo, t)
sys_promo = mtkcompile(bass_promo_sys)
println("  ✓ MTK model compiled")

# ============================================================
# Cell 9: MTK simulation
# ============================================================
println("\n[Cell 9] MTK simulation...")
M_est_rough = Float64(N_cum_obs[end]) * 1.3
tspan_sim   = (1.0, Float64(T_PERIODS))

prob_test = ODEProblem(sys_promo,
    Dict(N_bass => 0.0, p_bass => 0.005, q_bass => 0.05,
         M_bass => M_est_rough, gp_param => 1.0),
    tspan_sim)
sol_test = solve(prob_test, Tsit5(); saveat=1.0)
println("  ODE solved: $(length(sol_test.t)) time points, N_final=$(round(sol_test[N_bass][end], digits=1))")
println("  ✓ MTK simulation works")

# ============================================================
# Cell 11: Turing full model definition
# ============================================================
println("\n[Cell 11] Turing full model definition...")
@model function bass_nbd_model(
    n_trial  :: Vector{Int},
    n_repeat :: Vector{Int},
    promo    :: Vector{Int},
    M_lo     :: Float64,
    M_prior  :: Float64
)
    T = length(n_trial)

    p_v     ~ truncated(Normal(0.010, 0.010), 1e-5, 0.3)
    q_v     ~ truncated(Normal(0.050, 0.050), 1e-5, 1.0)
    M_v     ~ truncated(Normal(M_prior, 1000.0), M_lo, M_prior * 3.0)
    gamma_p ~ Normal(0.0, 1.0)
    sigma_t ~ Exponential(5.0)
    K_r     ~ truncated(Normal(0.80, 0.50), 0.05, 10.0)
    mu_r    ~ truncated(Normal(0.50, 0.30), 0.01, 5.0)
    delta_r ~ Normal(0.0, 1.0)

    N_cum      = zero(p_v)
    lambda_vec = Vector{typeof(p_v)}(undef, T)
    N_prev_vec = Vector{typeof(p_v)}(undef, T)

    for t in 1:T
        N_prev_vec[t] = N_cum
        raw_lam = (p_v * exp(gamma_p * promo[t]) + q_v * N_cum / M_v) *
                  (M_v - N_cum)
        lam_t = raw_lam > 0 ? raw_lam : zero(raw_lam)
        lambda_vec[t] = lam_t
        N_cum = N_cum + lam_t
    end

    for t in 1:T
        mu_t = lambda_vec[t] + 1e-3 * one(p_v)
        n_trial[t] ~ Normal(mu_t, sigma_t + sqrt(mu_t))
    end

    for t in 2:T
        N_prev = N_prev_vec[t]
        if N_prev > 0.5
            mu_r_t = mu_r * exp(delta_r * promo[t])
            p_nb   = K_r / (K_r + mu_r_t)
            r_nb   = N_prev * K_r
            n_repeat[t] ~ NegativeBinomial(r_nb, p_nb; check_args=false)
        end
    end
end
println("  ✓ Full model defined")

# ============================================================
# Cell 12: Null model definition
# ============================================================
println("\n[Cell 12] Null model definition...")
@model function bass_nbd_model_null(
    n_trial  :: Vector{Int},
    n_repeat :: Vector{Int},
    M_lo     :: Float64,
    M_prior  :: Float64
)
    T = length(n_trial)

    p_v     ~ truncated(Normal(0.010, 0.010), 1e-5, 0.3)
    q_v     ~ truncated(Normal(0.050, 0.050), 1e-5, 1.0)
    M_v     ~ truncated(Normal(M_prior, 1000.0), M_lo, M_prior * 3.0)
    sigma_t ~ Exponential(5.0)
    K_r     ~ truncated(Normal(0.80, 0.50), 0.05, 10.0)
    mu_r    ~ truncated(Normal(0.50, 0.30), 0.01, 5.0)

    N_cum      = zero(p_v)
    lambda_vec = Vector{typeof(p_v)}(undef, T)
    N_prev_vec = Vector{typeof(p_v)}(undef, T)

    for t in 1:T
        N_prev_vec[t] = N_cum
        raw_lam = (p_v + q_v * N_cum / M_v) * (M_v - N_cum)
        lam_t = raw_lam > 0 ? raw_lam : zero(raw_lam)
        lambda_vec[t] = lam_t
        N_cum = N_cum + lam_t
    end

    for t in 1:T
        mu_t = lambda_vec[t] + 1e-3 * one(p_v)
        n_trial[t] ~ Normal(mu_t, sigma_t + sqrt(mu_t))
    end

    for t in 2:T
        N_prev = N_prev_vec[t]
        if N_prev > 0.5
            p_nb = K_r / (K_r + mu_r)
            r_nb = N_prev * K_r
            n_repeat[t] ~ NegativeBinomial(r_nb, p_nb; check_args=false)
        end
    end
end
println("  ✓ Null model defined")

# ============================================================
# Cell 14: MCMC - Full model (reduced samples for testing)
# ============================================================
println("\n[Cell 14] MCMC Full model (test: 200 samples)...")
M_LO    = Float64(N_cum_obs[end]) * 1.05
M_PRIOR = Float64(N_cum_obs[end]) * 1.30
@printf("  M prior: lo=%.0f  center=%.0f\n", M_LO, M_PRIOR)

model_full = bass_nbd_model(
    Int.(n_trial_obs), Int.(n_repeat_obs), Int.(promo_vec), M_LO, M_PRIOR
)

@time chain_full = sample(
    model_full,
    NUTS(200, 0.65),
    MCMCSerial(), 400, 2;
    progress = false
)
println("  chain size: ", size(chain_full))
println("  ✓ Full model MCMC completed")

# ============================================================
# Cell 15: MCMC - Null model
# ============================================================
println("\n[Cell 15] MCMC Null model (test: 200 samples)...")
model_null = bass_nbd_model_null(
    Int.(n_trial_obs), Int.(n_repeat_obs), M_LO, M_PRIOR
)

@time chain_null = sample(
    model_null,
    NUTS(200, 0.65),
    MCMCSerial(), 400, 2;
    progress = false
)
println("  chain size: ", size(chain_null))
println("  ✓ Null model MCMC completed")

# ============================================================
# Cell 17: Convergence diagnostics
# ============================================================
println("\n[Cell 17] Convergence diagnostics...")
summarize(chain_full)
println("  ✓ Diagnostics displayed")

# ============================================================
# Cell 20: Posterior analysis
# ============================================================
println("\n[Cell 20] Posterior analysis...")
gamma_p_samp = vec(chain_full[:gamma_p])
delta_r_samp = vec(chain_full[:delta_r])

prob_gp = mean(gamma_p_samp .> 0)
prob_dr = mean(delta_r_samp .> 0)
ci_gp   = quantile(gamma_p_samp, [0.025, 0.975])
ci_dr   = quantile(delta_r_samp, [0.025, 0.975])

@printf("  γp: mean=%.3f  CI=[%.3f, %.3f]  P(>0)=%.3f\n",
    mean(gamma_p_samp), ci_gp[1], ci_gp[2], prob_gp)
@printf("  δr: mean=%.3f  CI=[%.3f, %.3f]  P(>0)=%.3f\n",
    mean(delta_r_samp), ci_dr[1], ci_dr[2], prob_dr)
println("  ✓ Posterior analysis done")

# ============================================================
# Cell 22: Judgment table
# ============================================================
println("\n[Cell 22] Judgment table...")
function judge(prob_pos::Float64, ci_lo::Float64, ci_hi::Float64)::String
    ci_lo > 0     && return "有意な促進効果あり"
    ci_hi < 0     && return "抑制効果の可能性"
    prob_pos > 0.90 && return "効果の傾向あり"
    return "効果は不明確"
end

function pstat(chain, sym)
    v = vec(chain[sym])
    (mean=mean(v), lo=quantile(v,0.025), hi=quantile(v,0.975))
end

df_judgment = DataFrame(
    効果        = ["試用促進 (γₚ)", "リピート促進 (δᵣ)"],
    事後平均    = round.([mean(gamma_p_samp), mean(delta_r_samp)], digits=3),
    CI95_下限   = round.([ci_gp[1], ci_dr[1]], digits=3),
    CI95_上限   = round.([ci_gp[2], ci_dr[2]], digits=3),
    P_効果あり  = round.([prob_gp, prob_dr], digits=3),
    倍率_推定   = round.(exp.([mean(gamma_p_samp), mean(delta_r_samp)]), digits=2),
    判定        = [judge(prob_gp, ci_gp[1], ci_gp[2]),
                   judge(prob_dr, ci_dr[1], ci_dr[2])]
)
display(df_judgment)
println("  ✓ Judgment table created")

# ============================================================
# Cell 24: Posterior predictive
# ============================================================
println("\n[Cell 24] Posterior predictive check...")
p_samp   = vec(chain_full[:p_v])
q_samp   = vec(chain_full[:q_v])
M_samp   = vec(chain_full[:M_v])
gp_samp_v = vec(chain_full[:gamma_p])
Kr_samp  = vec(chain_full[:K_r])
mur_samp = vec(chain_full[:mu_r])
dr_samp  = vec(chain_full[:delta_r])

n_post = length(p_samp)
T = T_PERIODS
trial_pred  = zeros(n_post, T)
repeat_pred = zeros(n_post, T)

for i in 1:n_post
    local Nc = 0.0
    for tt in 1:T
        lam = (p_samp[i] * exp(gp_samp_v[i] * promo_vec[tt]) +
               q_samp[i] * Nc / M_samp[i]) * max(M_samp[i] - Nc, 0.0)
        lam = max(lam, 0.001)
        trial_pred[i, tt] = lam
        if Nc > 0.5
            repeat_pred[i, tt] = Nc * mur_samp[i] * exp(dr_samp[i] * promo_vec[tt])
        end
        Nc += lam
    end
end

t_med = [median(trial_pred[:, j]) for j in 1:T]
println("  trial_pred median range: $(round(minimum(t_med), digits=1)) ~ $(round(maximum(t_med), digits=1))")
println("  ✓ Posterior predictive computed")

# ============================================================
# Cell 25: Full vs Null comparison
# ============================================================
println("\n[Cell 25] Model comparison...")
p_null_v = mean(vec(chain_null[:p_v]))
q_null_v = mean(vec(chain_null[:q_v]))
M_null_v = mean(vec(chain_null[:M_v]))

trial_null = zeros(T)
let Nc_null = 0.0
    for tt in 1:T
        lam = (p_null_v + q_null_v * Nc_null / M_null_v) * max(M_null_v - Nc_null, 0.0)
        lam = max(lam, 0.001)
        trial_null[tt] = lam
        Nc_null += lam
    end
end
println("  Null model trial range: $(round(minimum(trial_null), digits=1)) ~ $(round(maximum(trial_null), digits=1))")
println("  ✓ Model comparison computed")

# ============================================================
# Cell 27: NBD-Dirichlet data prep
# ============================================================
println("\n[Cell 27] NBD-Dirichlet data prep...")
brand_cols = names(df_brand)[3:end]
J = length(brand_cols)
n_cat_data    = Int.(df_brand.n_cat)
brand_mat     = Matrix{Int}(df_brand[:, brand_cols])

buyers_idx   = findall(n_cat_data .> 0)
n_cat_buyers = n_cat_data[buyers_idx]
brand_buyers = brand_mat[buyers_idx, :]
N_BUYERS     = length(buyers_idx)

share_obs = vec(sum(brand_buyers, dims=1)) ./ sum(brand_buyers)
pen_obs   = vec(sum(brand_buyers .> 0, dims=1)) ./ N_BUYERS
println("  Buyers: $N_BUYERS, Brands: $J")
println("  ✓ Dirichlet data ready")

# ============================================================
# Cell 28: Dirichlet model definition
# ============================================================
println("\n[Cell 28] Dirichlet model definition...")
@model function nbd_dirichlet_model(
    n_cat::Vector{Int},
    brand_counts::Matrix{Int},
    ::Val{J}
) where {J}
    N = length(n_cat)

    M_d ~ truncated(Normal(5.0, 3.0), 0.01, 30.0)
    K_d ~ truncated(Normal(1.0, 0.8), 0.01, 15.0)
    S_d ~ truncated(Normal(1.5, 1.0), 0.01, 15.0)
    p_d ~ Dirichlet(J, 1.0)

    alpha_d = S_d .* p_d
    p_nb_d  = K_d / (K_d + M_d)

    for i in 1:N
        n_cat[i] ~ NegativeBinomial(K_d, p_nb_d; check_args=false)
        if n_cat[i] > 0
            brand_counts[i, :] ~ DirichletMultinomial(n_cat[i], alpha_d)
        end
    end
end
println("  ✓ Dirichlet model defined")

# ============================================================
# Cell 29: Dirichlet MCMC
# ============================================================
println("\n[Cell 29] Dirichlet MCMC (subsample, test: 200 samples)...")
n_sub = min(N_BUYERS, 1000)
sub_idx = sort(Random.shuffle(MersenneTwister(42), 1:N_BUYERS)[1:n_sub])
model_dir = nbd_dirichlet_model(n_cat_buyers[sub_idx], brand_buyers[sub_idx, :], Val(J))

@time chain_dir = sample(
    model_dir,
    NUTS(200, 0.65),
    MCMCSerial(), 400, 2;
    progress = false
)
println("  ✓ Dirichlet MCMC completed")

# ============================================================
# Cell 30: Dirichlet results
# ============================================================
println("\n[Cell 30] Dirichlet results...")
M_d_est = mean(get(chain_dir, :M_d).M_d)
K_d_est = mean(get(chain_dir, :K_d).K_d)
S_d_est = mean(get(chain_dir, :S_d).S_d)

p_syms  = namesingroup(chain_dir, :p_d)
p_vals  = get(chain_dir, p_syms)
p_d_est = [mean(p_vals[s]) for s in p_syms]

@printf("  M_d=%.3f  K_d=%.3f  S_d=%.3f\n", M_d_est, K_d_est, S_d_est)
println("  Share estimates: ", round.(p_d_est, digits=3))
println("  ✓ Dirichlet results extracted")

# ============================================================
# Cell 32: Final summary
# ============================================================
println("\n" * "=" ^ 60)
println("  FINAL SUMMARY")
println("=" ^ 60)

@printf("\n  BASS: p=%.4f  q=%.4f  M=%.0f\n",
    pstat(chain_full, :p_v).mean,
    pstat(chain_full, :q_v).mean,
    pstat(chain_full, :M_v).mean)
@printf("  Promo trial effect  γp = %.3f  [%.3f, %.3f]  P(>0)=%.3f  ×%.2f\n",
    mean(gamma_p_samp), ci_gp[1], ci_gp[2], prob_gp, exp(mean(gamma_p_samp)))
@printf("  Promo repeat effect δr = %.3f  [%.3f, %.3f]  P(>0)=%.3f  ×%.2f\n",
    mean(delta_r_samp), ci_dr[1], ci_dr[2], prob_dr, exp(mean(delta_r_samp)))
@printf("  Dirichlet: M_d=%.2f  K_d=%.2f  S_d=%.2f\n", M_d_est, K_d_est, S_d_est)

println("\n  ALL CELLS VERIFIED SUCCESSFULLY ✓")
println("=" ^ 60)
