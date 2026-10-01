# An implementation of MINRES-QLP.
#
# This method is described in
#
# S.-C. T. Choi, Iterative methods for singular linear equations and least-squares problems.
# Ph.D. thesis, ICME, Stanford University, 2006.
#
# S.-C. T. Choi, C. C. Paige and M. A. Saunders, MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems.
# SIAM Journal on Scientific Computing, Vol. 33(4), pp. 1810--1836, 2011.
#
# S.-C. T. Choi and M. A. Saunders, Algorithm 937: MINRES-QLP for symmetric and Hermitian linear equations and least-squares problems.
# ACM Transactions on Mathematical Software, 40(2), pp. 1--12, 2014.
# 
# Negative curvature detection follows
# Liu, Yang, and Roosta, MINRES: from negative curvature detection to monotonicity properties,
# SIAM Journal on Optimization, 32(4), pp. 2636--2661, 2022.
#
# The residual and A-residual estimates of the NELQ iterate, as well as the
# Gauss-Radau upper bounds on the NELQ and MINRES-QLP errors, follow
# O. Mouhtal and D. Orban, NELQ: a minimum error iterative solver for symmetric
# linear systems with MINRES error control, Cahier du GERAD, 2026.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, September 2019.

export minres_qlp, minres_qlp!

"""
    (x, stats) = minres_qlp(A, b::AbstractVector{FC};
                            M=I, ldiv::Bool=false, Artol::T=√eps(T),
                            linesearch::Bool=false, λ::T=zero(T), atol::T=√eps(T),
                            rtol::T=√eps(T), nelq_residuals::Bool=false,
                            λest::T=zero(T), etol::T=√eps(T), itmax::Int=0,
                            timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                            callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = minres_qlp(A, b, x0::AbstractVector; kwargs...)

MINRES-QLP can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

MINRES-QLP is the only method based on the Lanczos process that returns the minimum-norm
solution on singular inconsistent systems (A + λI)x = b of size n, where λ is a shift parameter.
It is significantly more complex but can be more reliable than MINRES when A is ill-conditioned.

M also indicates the weighted norm in which residuals are measured.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :minres_qlp`.

For an in-place variant that reuses memory across solves, see [`minres_qlp!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `λ`: regularization parameter;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `Artol`: relative stopping tolerance based on the Aᴴ-residual norm;
* `nelq_residuals`: if `true`, compute the residual norm and the A-residual norm of the NELQ iterate;
* `λest`: positive strict lower bound on the smallest nonzero eigenvalue of A. If `λest > 0`, Gauss-Radau upper bounds on the NELQ and MINRES-QLP errors are computed. They are only valid when A is positive semidefinite, the system is consistent, and 0 < λest < λᵣ, where λᵣ is the smallest nonzero eigenvalue of A;
* `etol`: stopping tolerance based on the upper bound on the MINRES-QLP error;
* `linesearch`: if `true`, indicate that the solution is to be used in an inexact Newton method with linesearch. If `true` and nonpositive curvature is detected, the behavior depends on the iteration:
 – at iteration k = 1, the solver takes the right-hand side (i.e., the preconditioned negative gradient) as the current solution. The same search direction is returned in `workspace.npc_dir`, and `stats.npcCount` is set to 1;
 – at iteration k > 1, the solver returns the solution from iteration k – 1, the residual from iteration k is a nonpositive curvature direction stored in `stats.npc_dir` and `stats.npcCount` is set to 1;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`MinresQlpStats`](@ref) structure.

#### Stats

If `history = true`:

* `stats.residuals` and `stats.Aresiduals` contain the residual and A-residual norms of the MINRES-QLP iterates;
* `stats.residuals_nelq` contains the residual norms of the NELQ iterates (only if `nelq_residuals = true`);
* `stats.Aresiduals_nelq` contains the A-residual norms of the NELQ iterates (only if `nelq_residuals = true`);
* `stats.err_ubnds_nelq` contains upper bounds on the NELQ error for k ≥ 3 (only if `λest > 0`);
* `stats.err_ubnds_qlp` contains upper bounds on the MINRES-QLP error for k ≥ 3 (only if `λest > 0`);

When a preconditioner `M` is used, the errors are measured in the M⁻¹-norm.

#### References

* S.-C. T. Choi, *Iterative methods for singular linear equations and least-squares problems*, Ph.D. thesis, ICME, Stanford University, 2006.
* S.-C. T. Choi, C. C. Paige and M. A. Saunders, [*MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems*](https://doi.org/10.1137/100787921), SIAM Journal on Scientific Computing, Vol. 33(4), pp. 1810--1836, 2011.
* S.-C. T. Choi and M. A. Saunders, [*Algorithm 937: MINRES-QLP for symmetric and Hermitian linear equations and least-squares problems*](https://doi.org/10.1145/2527267), ACM Transactions on Mathematical Software, 40(2), pp. 1--12, 2014.
* Y. Liu and F. Roosta, [*MINRES: From Negative Curvature Detection to Monotonicity Properties*](https://doi.org/10.1137/21M143666X), SIAM Journal on Optimization, 32(4), pp. 2636--2661, 2022.
* O. Mouhtal and D. Orban, *NELQ: A minimum error iterative solver for symmetric linear systems with MINRES error control*, Cahier du GERAD, 2026.

"""
function minres_qlp end

"""
    workspace = minres_qlp!(workspace::MinresQlpWorkspace, A, b; kwargs...)
    workspace = minres_qlp!(workspace::MinresQlpWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`minres_qlp`](@ref).

See [`MinresQlpWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :minres_qlp` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function minres_qlp! end

def_args_minres_qlp = (:(A                    ),
                       :(b::AbstractVector{FC}))

def_optargs_minres_qlp = (:(x0::AbstractVector),)

def_kwargs_minres_qlp = (:(; M = I                        ),
                         :(; ldiv::Bool = false           ),
                         :(; linesearch::Bool = false     ),
                         :(; λ::T = zero(T)               ),
                         :(; atol::T = √eps(T)            ),
                         :(; rtol::T = √eps(T)            ),
                         :(; Artol::T = √eps(T)           ),
                         :(; nelq_residuals::Bool = false ),
                         :(; λest::T = zero(T)            ),
                         :(; etol::T = √eps(T)            ),
                         :(; itmax::Int = 0               ),
                         :(; timemax::Float64 = Inf       ),
                         :(; verbose::Int = 0             ),
                         :(; history::Bool = false        ),
                         :(; callback = workspace -> false),
                         :(; iostream::IO = kstdout       ))

def_kwargs_minres_qlp = extract_parameters.(def_kwargs_minres_qlp)

args_minres_qlp = (:A, :b)
optargs_minres_qlp = (:x0,)
kwargs_minres_qlp = (:M, :ldiv, :linesearch, :λ, :atol, :rtol, :Artol, :nelq_residuals, :λest, :etol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function minres_qlp!(workspace :: MinresQlpWorkspace{T,FC,S}, $(def_args_minres_qlp...); $(def_kwargs_minres_qlp...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "MINRES-QLP: system of size %d\n", n)
    (workspace.warm_start && linesearch) && error("warm_start and linesearch cannot be used together")
    λest ≥ 0 || error("λest must be nonnegative")

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    # Set up workspace.
    allocate_if(!MisI, workspace, :vₖ, S, workspace.x)  # The length of vₖ is n
    allocate_if(linesearch, workspace, :npc_dir , S, workspace.x)  # The length of npc_dir is n
    if linesearch
      npc_dir = workspace.npc_dir
    end
    wₖ₋₁, wₖ, M⁻¹vₖ₋₁, M⁻¹vₖ = workspace.wₖ₋₁, workspace.wₖ, workspace.M⁻¹vₖ₋₁, workspace.M⁻¹vₖ
    Δx, x, p, stats = workspace.Δx, workspace.x, workspace.p, workspace.stats
    warm_start = workspace.warm_start
    rNorms, ArNorms, Aconds = stats.residuals, stats.Aresiduals, stats.Acond
    rNorms_nelq, ArNorms_nelq = stats.residuals_nelq, stats.Aresiduals_nelq
    err_ubnds_nelq, err_ubnds_qlp = stats.err_ubnds_nelq, stats.err_ubnds_qlp
    reset!(stats)
    vₖ = MisI ? M⁻¹vₖ : workspace.vₖ
    vₖ₊₁ = MisI ? p : M⁻¹vₖ₋₁

    # Initial solution x₀
    kfill!(x, zero(FC))

    if warm_start
      kmul!(M⁻¹vₖ, A, Δx)
      (λ ≠ 0) && kaxpy!(n, λ, Δx, M⁻¹vₖ)
      kaxpby!(n, one(FC), b, -one(FC), M⁻¹vₖ)
    else
      kcopy!(n, M⁻¹vₖ, b)  # M⁻¹vₖ ← b
    end

    # β₁v₁ = Mb
    MisI || mulorldiv!(vₖ, M, M⁻¹vₖ, ldiv)
    βₖ = knorm_elliptic(n, vₖ, M⁻¹vₖ)
    linesearch && kcopy!(n, npc_dir , vₖ)  # npc_dir  ← v; contain the preconditioned initial residual
    if βₖ ≠ 0
      kdiv!(n, M⁻¹vₖ, βₖ)
      MisI || kdiv!(n, vₖ, βₖ)
    end

    rNorm = βₖ
    ANorm² = zero(T)
    ANorm = zero(T)
    μmin = zero(T)
    μmax = zero(T)
    Acond = zero(T)
    history && push!(rNorms, rNorm)
    history && push!(Aconds, Acond)
    (history && nelq_residuals) && push!(rNorms_nelq, rNorm)  # ‖rᴺ₀‖ = β₁
    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    ε = atol + rtol * rNorm
    κ = zero(T)
    if nelq_residuals
      (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %9s  %7s  %7s  %8s  %7s  %7s  %8s  %5s\n", "k", "‖rₖ‖", "‖Arₖ₋₁‖", "‖rᴺₖ‖", "‖Arᴺₖ₋₁‖", "βₖ₊₁", "Rₖ.ₖ", "Lₖ.ₖ", "‖A‖", "κ(A)", "backward", "timer")
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7s  %7.1e  %9s  %7.1e  %7s  %8s  %7.1e  %7.1e  %8s  %.2fs\n", iter, rNorm, "✗ ✗ ✗ ✗", rNorm, "✗ ✗ ✗ ✗", βₖ, "✗ ✗ ✗ ✗", " ✗ ✗ ✗ ✗", ANorm, Acond, " ✗ ✗ ✗ ✗", start_time |> ktimer)
    else
      (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %7s  %8s  %7s  %7s  %8s  %5s\n", "k", "‖rₖ‖", "‖Arₖ₋₁‖", "βₖ₊₁", "Rₖ.ₖ", "Lₖ.ₖ", "‖A‖", "κ(A)", "backward", "timer")
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7s  %7.1e  %7s  %8s  %7.1e  %7.1e  %8s  %.2fs\n", iter, rNorm, "✗ ✗ ✗ ✗", βₖ, "✗ ✗ ✗ ✗", " ✗ ✗ ✗ ✗", ANorm, Acond, " ✗ ✗ ✗ ✗", start_time |> ktimer)
    end

    # Set up workspace.
    kfill!(M⁻¹vₖ₋₁, zero(FC))
    ζbarₖ = βₖ
    ξₖ₋₁ = zero(T)
    τₖ₋₂ = τₖ₋₁ = τₖ = zero(T)
    ψbarₖ₋₂ = zero(T)
    μbisₖ₋₂ = μbarₖ₋₁ = zero(T)
    kfill!(wₖ₋₁, zero(FC))
    kfill!(wₖ, zero(FC))
    cₖ₋₂ = cₖ₋₁ = cₖ = one(T)   # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
    sₖ₋₂ = sₖ₋₁ = sₖ = zero(T)  # Givens sines used for the QR factorization of Tₖ₊₁.ₖ

    # NELQ iterate xᴺₖ = Vₖyᴺₖ.
    # In the loop, x contains xᴺₖ and the MINRES-QLP iterate is xₖ = xᴺₖ + τₖ₋₁wₖ₋₁ + τₖwₖ.
    rNorm_nelq = rNorm           # ‖rᴺₖ‖
    ArNorm_nelq = zero(T)        # ‖Arᴺₖ₋₁‖
    AbNorm = zero(T)             # ‖Ab‖ = ‖Arᴺₖ‖ for k ≤ 2 because xᴺ₀ = xᴺ₁ = xᴺ₂ = 0
    χoldₖ₋₂ = χoldₖ₋₁ = zero(T)  # last two components of tₖ₋₁ - Rₖ₋₁yᴺₖ₋₁
    νoldₖ₋₂ = νoldₖ₋₁ = zero(T)  # last two components of yᴺₖ₋₁
    κₖ₋₁ = zero(T)               # coefficient of vₖ₋₁ in ẘₖ₋₂
    λₖ₋₂ = λₖ₋₁ = zero(T)        # diagonal entries of Rₖ₋₁
    γₖ₋₂ = zero(T)               # superdiagonal entry of Rₖ₋₁
    αₖ₋₁ = βₖ₋₁ = zero(T)        # Lanczos coefficients of the previous iteration
    χₖ₋₁ = χₖ = zero(T)          # last two components of tₖ - Rₖyᴺₖ
    νₖ₋₁ = νₖ = zero(T)          # last two components of yᴺₖ
    κₖ = zero(T)                 # coefficient of vₖ in ẘₖ₋₁
    xNorm²_nelq = zero(T)        # ‖xᴺₖ‖²

    # Gauss-Radau upper bounds on the NELQ and MINRES-QLP errors.
    error_bnd = λest > 0
    # QR factorization of Tₖ - λest I = Q̃ₖR̃ₖ with symmetric Givens reflections, as in SYMMLQ.
    ρbarʷ = zero(T)          # last diagonal entry of R̃ₖ before the next reflection
    σbarʷ = zero(T)          # entry (k, k+1) of the partially reduced Tₖ₊₁ - λest I
    cʷₖ₋₁ = sʷₖ₋₁ = zero(T)  # last reflection
    cʷₖ₋₂ = -one(T)          # previous reflection (c₀ = -1 by convention)
    ωₖ = zero(T)             # ωₖ is such that λₘᵢₙ(T̃ₖ) = λest where T̃ₖ is Tₖ with αₖ replaced by ωₖ
    err_ubnd_qlp = zero(T)
    fwd_err_ubnd = false

    # Tolerance for breakdown detection.
    btol = eps(T)^(3/4)

    # Stopping criterion.
    breakdown = false
    solved = zero_resid = zero_resid_lim = rNorm ≤ ε
    zero_resid_mach = false
    inconsistent = false
    ill_cond_mach = false
    tired = iter ≥ itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false
    stats.indefinite = false

    while !(solved || tired || inconsistent || ill_cond_mach || breakdown || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the preconditioned Lanczos process.
      # M(A + λI)Vₖ = Vₖ₊₁Tₖ₊₁.ₖ
      # βₖ₊₁vₖ₊₁ = M(A + λI)vₖ - αₖvₖ - βₖvₖ₋₁

      kmul!(p, A, vₖ)        # p ← Avₖ
      if λ ≠ 0
        kaxpy!(n, λ, vₖ, p)  # p ← p + λvₖ
      end

      if iter ≥ 2
        kaxpy!(n, -βₖ, M⁻¹vₖ₋₁, p)  # p ← p - βₖ * M⁻¹vₖ₋₁
      end

      αₖ = kdotr(n, vₖ, p)  # αₖ = ⟨vₖ,p⟩

      kaxpy!(n, -αₖ, M⁻¹vₖ, p)  # p ← p - αₖM⁻¹vₖ

      MisI || mulorldiv!(vₖ₊₁, M, p, ldiv)  # βₖ₊₁vₖ₊₁ = MAvₖ - γₖvₖ₋₁ - αₖvₖ

      βₖ₊₁ = knorm_elliptic(m, vₖ₊₁, p)

      # βₖ₊₁.ₖ ≠ 0
      if βₖ₊₁ > btol
        kdiv!(m, vₖ₊₁, βₖ₊₁)
        MisI || kdiv!(m, p, βₖ₊₁)
      end

      ANorm² = ANorm² + αₖ * αₖ + βₖ * βₖ + βₖ₊₁ * βₖ₊₁

      # Update ‖Arᴺₖ₋₁‖ estimate.
      # It requires αₖ and βₖ₊₁, hence the delay of one iteration.
      if nelq_residuals
        if iter == 1
          # ‖Ab‖ = β₁‖T₂.₁e₁‖
          AbNorm = βₖ * sqrt(αₖ * αₖ + βₖ₊₁ * βₖ₊₁)
        end
        if iter ≤ 2
          ArNorm_nelq = AbNorm
        else
          # ‖Arᴺₖ₋₁‖² = (λₖ₋₂χₖ₋₂)² + (γₖ₋₂χₖ₋₂ + λₖ₋₁χₖ₋₁)² + βₖ²(θᴺₖ₋₁² + (βₖ₊₁νₖ₋₁)²)
          # with θᴺₖ₋₁ = βₖ₋₁νₖ₋₂ + (αₖ₋₁ + αₖ)νₖ₋₁, where νₖ₋₂ and νₖ₋₁ are the last two components of yᴺₖ₋₁.
          θᴺₖ₋₁ = βₖ₋₁ * νoldₖ₋₂ + (αₖ₋₁ + αₖ) * νoldₖ₋₁
          ArNorm_nelq = sqrt(abs2(λₖ₋₂ * χoldₖ₋₂) + abs2(γₖ₋₂ * χoldₖ₋₂ + λₖ₋₁ * χoldₖ₋₁) +
                             βₖ * βₖ * (θᴺₖ₋₁ * θᴺₖ₋₁ + abs2(βₖ₊₁ * νoldₖ₋₁)))
        end
        history && push!(ArNorms_nelq, ArNorm_nelq)
      end

      # Gauss-Radau parameter ωₖ = λest + βₖ² (eₖ₋₁)ᵀ(Tₖ₋₁ - λest I)⁻¹eₖ₋₁ = λest - βₖ² cʷₖ₋₂ / ρbarʷₖ₋₁,
      # computed from the QR factorization of Tₖ - λest I.
      if error_bnd
        if iter == 1
          # [cʷ₁  sʷ₁] [α₁ - λest] = [ρʷ₁]
          # [sʷ₁ -cʷ₁] [   β₂    ]   [ 0 ]
          ρbarʷ = αₖ - λest
          σbarʷ = βₖ₊₁
        else
          ωₖ = λest - βₖ * βₖ * cʷₖ₋₂ / ρbarʷ
          # Apply the reflection Q̃ₖ₋₁.ₖ to the last column of Tₖ₊₁ - λest I
          ρbarʷ = sʷₖ₋₁ * σbarʷ - cʷₖ₋₁ * (αₖ - λest)
          σbarʷ = -cʷₖ₋₁ * βₖ₊₁
          cʷₖ₋₂ = cʷₖ₋₁
        end
        ρʷ = sqrt(ρbarʷ * ρbarʷ + βₖ₊₁ * βₖ₊₁)
        if ρʷ > 0
          cʷₖ₋₁ = ρbarʷ / ρʷ
          sʷₖ₋₁ = βₖ₊₁ / ρʷ
        end
      end

      # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
      #                                            [ Oᵀ ]
      #
      # [ α₁ β₂ 0  •  •  •   0  ]      [ λ₁ γ₁ ϵ₁ 0  •  •  0  ]
      # [ β₂ α₂ β₃ •         •  ]      [ 0  λ₂ γ₂ •  •     •  ]
      # [ 0  •  •  •  •      •  ]      [ •  •  λ₃ •  •  •  •  ]
      # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
      # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
      # [ •        •  •  •   βₖ ]      [ •           •  • γₖ₋₁]
      # [ •           •  βₖ  αₖ ]      [ 0  •  •  •  •  0  λₖ ]
      # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
      #
      # If k = 1, we don't have any previous reflection.
      # If k = 2, we apply the last reflection.
      # If k ≥ 3, we only apply the two previous reflections.

      # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
      if iter ≥ 3
        # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
        # [sₖ₋₂ -cₖ₋₂] [βₖ]   [γbarₖ₋₁]
        ϵₖ₋₂    =  sₖ₋₂ * βₖ
        γbarₖ₋₁ = -cₖ₋₂ * βₖ
      end
      # Apply previous Givens reflections Qₖ₋₁.ₖ
      if iter ≥ 2
        iter == 2 && (γbarₖ₋₁ = βₖ)
        # [cₖ₋₁  sₖ₋₁] [γbarₖ₋₁] = [γₖ₋₁ ]
        # [sₖ₋₁ -cₖ₋₁] [   αₖ  ]   [λbarₖ]
        γₖ₋₁  = cₖ₋₁ * γbarₖ₋₁ + sₖ₋₁ * αₖ
        λbarₖ = sₖ₋₁ * γbarₖ₋₁ - cₖ₋₁ * αₖ
      end
      iter == 1 && (λbarₖ = αₖ)

      # Check for nonpositive curvature
      if linesearch
        # if the first iteration, we need to check c₁ * λbar₁ = -α₁
        if iter == 1
          cγ = -αₖ
        else
          cγ = cₖ₋₁ * λbarₖ
        end
        if cγ ≥ 0
          if iter == 1
            kcopy!(n, x, b)
          end
          # Nonpositive curvature detected.
          (verbose > 0) && @printf(iostream, "nonpositive curvature detected:  cₖ * λbarₖ = %e\n", cγ)
          stats.solved = true
          stats.npcCount = 1
          stats.niter = iter
          stats.inconsistent = false
          stats.timer = start_time |> ktimer
          stats.status = "nonpositive curvature"
          workspace.warm_start = false
          stats.indefinite = true
          return workspace
        end
      end

      # Compute and apply current Givens reflection Qₖ.ₖ₊₁
      # [cₖ  sₖ] [λbarₖ] = [λₖ]
      # [sₖ -cₖ] [βₖ₊₁ ]   [0 ]
      (cₖ, sₖ, λₖ) = sym_givens(λbarₖ, βₖ₊₁)

      # Compute z̅ₖ₊₁ = [   zₖ  ] = (Qₖ)ᴴβ₁e₁
      #                [ζbarₖ₊₁]
      #
      # [cₖ  sₖ] [ζbarₖ] = [   ζₖ  ]
      # [sₖ -cₖ] [  0  ]   [ζbarₖ₊₁]
      ζₖ      = cₖ * ζbarₖ
      ζbarₖ₊₁ = sₖ * ζbarₖ

      # check for nonpositive curvature
      if linesearch
        kscal!(n, sₖ * sₖ, npc_dir)  # npc_dir  = sₖ * sₖ * npc_dir
        kaxpy!(n, -ζbarₖ₊₁ * cₖ , vₖ₊₁, npc_dir)  # npc_dir  = npc_dir  - ζbarₖ * cₖ * vₖ, this is rₖ
      end
      
      # Update the LQ factorization of Rₖ = LₖPₖ.
      # [ λ₁ γ₁ ϵ₁ 0  •  •  0  ]   [ μ₁   0    •    •     •      •      0  ]
      # [ 0  λ₂ γ₂ •  •     •  ]   [ ψ₁   μ₂   •                        •  ]
      # [ •  •  λ₃ •  •  •  •  ]   [ ρ₁   ψ₂   μ₃   •                   •  ]
      # [ •     •  •  •  •  0  ] = [ 0    •    •    •     •             •  ] Pₖ
      # [ •        •  •  • ϵₖ₋₂]   [ •    •    •    •   μₖ₋₂     •      •  ]
      # [ •           •  • γₖ₋₁]   [ •         •    •   ψₖ₋₂  μbisₖ₋₁   0  ]
      # [ 0  •  •  •  •  0  λₖ ]   [ 0    •    •    0   ρₖ₋₂  ψbarₖ₋₁ μbarₖ]

      if iter == 1
        μbarₖ = λₖ
      elseif iter == 2
        # [μbar₁ γ₁] [cp₂  sp₂] = [μbis₁   0  ]
        # [  0   λ₂] [sp₂ -cp₂]   [ψbar₁ μbar₂]
        (cpₖ, spₖ, μbisₖ₋₁) = sym_givens(μbarₖ₋₁, γₖ₋₁)
        ψbarₖ₋₁ =  spₖ * λₖ
        μbarₖ   = -cpₖ * λₖ
      else
        # [μbisₖ₋₂   0     ϵₖ₋₂] [cpₖ  0   spₖ]   [μₖ₋₂   0     0 ]
        # [ψbarₖ₋₂ μbarₖ₋₁ γₖ₋₁] [ 0   1    0 ] = [ψₖ₋₂ μbarₖ₋₁ θₖ]
        # [  0       0      λₖ ] [spₖ  0  -cpₖ]   [ρₖ₋₂   0     ηₖ]
        (cpₖ, spₖ, μₖ₋₂) = sym_givens(μbisₖ₋₂, ϵₖ₋₂)
        ψₖ₋₂ =  cpₖ * ψbarₖ₋₂ + spₖ * γₖ₋₁
        θₖ   =  spₖ * ψbarₖ₋₂ - cpₖ * γₖ₋₁
        ρₖ₋₂ =  spₖ * λₖ
        ηₖ   = -cpₖ * λₖ

        # [μₖ₋₂   0     0 ] [1   0    0 ]   [μₖ₋₂   0       0  ]
        # [ψₖ₋₂ μbarₖ₋₁ θₖ] [0  cdₖ  sdₖ] = [ψₖ₋₂ μbisₖ₋₁   0  ]
        # [ρₖ₋₂   0     ηₖ] [0  sdₖ -cdₖ]   [ρₖ₋₂ ψbarₖ₋₁ μbarₖ]
        (cdₖ, sdₖ, μbisₖ₋₁) = sym_givens(μbarₖ₋₁, θₖ)
        ψbarₖ₋₁ =  sdₖ * ηₖ
        μbarₖ   = -cdₖ * ηₖ
      end

      # Compute Lₖtₖ = zₖ
      # [ μ₁   0    •    •     •      •      0  ] [τ₁]   [ζ₁]
      # [ ψ₁   μ₂   •                        •  ] [τ₂]   [ζ₂]
      # [ ρ₁   ψ₂   μ₃   •                   •  ] [τ₃]   [ζ₃]
      # [ 0    •    •    •     •             •  ] [••] = [••]
      # [ •    •    •    •   μₖ₋₂     •      •  ] [••]   [••]
      # [ •         •    •   ψₖ₋₂  μbisₖ₋₁   0  ] [••]   [••]
      # [ 0    •    •    0   ρₖ₋₂  ψbarₖ₋₁ μbarₖ] [τₖ]   [ζₖ]
      if iter == 1
        τₖ = ζₖ / μbarₖ
      elseif iter == 2
        τₖ₋₁ = τₖ
        τₖ₋₁ = τₖ₋₁ * μbarₖ₋₁ / μbisₖ₋₁
        ξₖ   = ζₖ
        τₖ   = (ξₖ - ψbarₖ₋₁ * τₖ₋₁) / μbarₖ
      else
        τₖ₋₂ = τₖ₋₁
        τₖ₋₂ = τₖ₋₂ * μbisₖ₋₂ / μₖ₋₂
        τₖ₋₁ = (ξₖ₋₁ - ψₖ₋₂ * τₖ₋₂) / μbisₖ₋₁
        ξₖ   = ζₖ - ρₖ₋₂ * τₖ₋₂
        τₖ   = (ξₖ - ψbarₖ₋₁ * τₖ₋₁) / μbarₖ
      end

      # Compute directions wₖ₋₂, ẘₖ₋₁ and w̄ₖ, last columns of Wₖ = Vₖ(Pₖ)ᴴ
      if iter == 1
        # w̅₁ = v₁
        kcopy!(n, wₖ, vₖ)
      elseif iter == 2
        # [w̅ₖ₋₁ vₖ] [cpₖ  spₖ] = [ẘₖ₋₁ w̅ₖ] ⟷ ẘₖ₋₁ = cpₖ * w̅ₖ₋₁ + spₖ * vₖ
        #           [spₖ -cpₖ]             ⟷ w̅ₖ   = spₖ * w̅ₖ₋₁ - cpₖ * vₖ
        @kswap!(wₖ₋₁, wₖ)
        # Compute w̅ₖ
        kcopy!(n, wₖ, wₖ₋₁)
        kaxpby!(n, -cpₖ, vₖ, spₖ, wₖ)
        # Compute ẘₖ₋₁
        kaxpby!(n, spₖ, vₖ, cpₖ, wₖ₋₁)
      else
        # [ẘₖ₋₂ w̄ₖ₋₁ vₖ] [cpₖ  0   spₖ] [1   0    0 ] = [wₖ₋₂ ẘₖ₋₁ w̄ₖ] ⟷ wₖ₋₂ = cpₖ * ẘₖ₋₂ + spₖ * vₖ
        #                [ 0   1    0 ] [0  cdₖ  sdₖ]                  ⟷ ẘₖ₋₁ = cdₖ * w̄ₖ₋₁ + sdₖ * (spₖ * ẘₖ₋₂ - cpₖ * vₖ)
        #                [spₖ  0  -cpₖ] [0  sdₖ -cdₖ]                  ⟷ w̄ₖ   = sdₖ * w̄ₖ₋₁ - cdₖ * (spₖ * ẘₖ₋₂ - cpₖ * vₖ)
        ẘₖ₋₂ = wₖ₋₁
        w̄ₖ₋₁ = wₖ
        # Update the solution x
        kaxpy!(n, cpₖ * τₖ₋₂, ẘₖ₋₂, x)
        kaxpy!(n, spₖ * τₖ₋₂, vₖ, x)
        # Compute wₐᵤₓ = spₖ * ẘₖ₋₂ - cpₖ * vₖ
        kaxpby!(n, -cpₖ, vₖ, spₖ, ẘₖ₋₂)
        wₐᵤₓ = ẘₖ₋₂
        # Compute ẘₖ₋₁ and w̄ₖ
        kref!(n, w̄ₖ₋₁, wₐᵤₓ, cdₖ, sdₖ)
        @kswap!(wₖ₋₁, wₖ)
      end

      # Update vₖ, M⁻¹vₖ₋₁, M⁻¹vₖ
      MisI || kcopy!(n, vₖ, vₖ₊₁)  # vₖ ← vₖ₊₁
      kcopy!(n, M⁻¹vₖ₋₁, M⁻¹vₖ)    # M⁻¹vₖ₋₁ ← M⁻¹vₖ
      kcopy!(n, M⁻¹vₖ, p)          # M⁻¹vₖ ← p

      # Update ‖rₖ‖ estimate
      # ‖ rₖ ‖ = |ζbarₖ₊₁|
      rNorm = abs(ζbarₖ₊₁)
      history && push!(rNorms, rNorm)

      # Update ‖Arₖ₋₁‖ estimate
      # ‖ Arₖ₋₁ ‖ = |ζbarₖ| * √(|λbarₖ|² + |γbarₖ|²)
      ArNorm = abs(ζbarₖ) * √(abs2(λbarₖ) + abs2(cₖ₋₁ * βₖ₊₁))
      iter == 1 && (κ = atol + Artol * ArNorm)
      history && push!(ArNorms, ArNorm)

      ANorm = sqrt(ANorm²)
      # estimate A condition number
      abs_μbarₖ = abs(μbarₖ)
      if iter == 1
        μmin = abs_μbarₖ
        μmax = abs_μbarₖ
      elseif iter == 2
        μmax = max(μmax, μbisₖ₋₁, abs_μbarₖ)
        μmin = min(μmin, μbisₖ₋₁, abs_μbarₖ)
      else
        μmax = max(μmax, μₖ₋₂, μbisₖ₋₁, abs_μbarₖ)
        μmin = min(μmin, μₖ₋₂, μbisₖ₋₁, abs_μbarₖ)
      end
      Acond = μmax / μmin
      history && push!(Aconds, Acond)

      # Update ‖xᴺₖ‖², xᴺₖ = xᴺₖ₋₁ + τₖ₋₂wₖ₋₂
      (iter ≥ 3) && (xNorm²_nelq += τₖ₋₂ * τₖ₋₂)

      # Update ‖rᴺₖ‖ estimate and the quantities needed by ‖Arᴺₖ‖ at the next iteration.
      if nelq_residuals
        # ‖rᴺₖ‖² = χₖ₋₁² + χₖ² + ζbarₖ₊₁²
        if iter == 1
          # xᴺ₁ = 0 and ‖rᴺ₁‖ = β₁
          χₖ₋₁ = zero(T)
          χₖ = zero(T)
        else
          χₖ₋₁ = μbisₖ₋₁ * τₖ₋₁  # τₖ₋₁ - ρₖ₋₃τₖ₋₃ - ψₖ₋₂τₖ₋₂
          χₖ = ξₖ                # τₖ - ρₖ₋₂τₖ₋₂
          rNorm_nelq = sqrt(χₖ₋₁ * χₖ₋₁ + χₖ * χₖ + ζbarₖ₊₁ * ζbarₖ₊₁)
        end
        history && push!(rNorms_nelq, rNorm_nelq)

        # Last two components νₖ₋₁ and νₖ of yᴺₖ, i.e., the coefficients of vₖ₋₁ and vₖ in xᴺₖ,
        # and coefficient κₖ of vₖ in ẘₖ₋₁.
        if iter == 1
          νₖ₋₁ = νₖ = κₖ = zero(T)
        elseif iter == 2
          νₖ₋₁ = νₖ = zero(T)
          κₖ = spₖ
        else
          νₖ₋₁ = νoldₖ₋₁ + cpₖ * κₖ₋₁ * τₖ₋₂
          νₖ = spₖ * τₖ₋₂
          κₖ = -sdₖ * cpₖ
        end
      end

      # Gauss-Radau upper bounds on the NELQ and MINRES-QLP errors.
      # The QLP factorization of T̃ₖ only differs from the one of Tₖ in its last two rows.
      if error_bnd && iter ≥ 3
        # [cₖ₋₁  sₖ₋₁] [γbarₖ₋₁] = [γ̃ₖ₋₁]
        # [sₖ₋₁ -cₖ₋₁] [   ωₖ  ]   [ λ̃ₖ ]
        γ̃ₖ₋₁ = cₖ₋₁ * γbarₖ₋₁ + sₖ₋₁ * ωₖ
        λ̃ₖ   = sₖ₋₁ * γbarₖ₋₁ - cₖ₋₁ * ωₖ
        # Apply the reflection Pₖ₋₂.ₖ
        ψ̃ₖ₋₂ = cpₖ * ψbarₖ₋₂ + spₖ * γ̃ₖ₋₁
        θ̃ₖ   = spₖ * ψbarₖ₋₂ - cpₖ * γ̃ₖ₋₁
        ρ̃ₖ₋₂ =  spₖ * λ̃ₖ
        η̃ₖ   = -cpₖ * λ̃ₖ
        # Compute and apply the reflection P̃ₖ₋₁.ₖ
        (c̃dₖ, s̃dₖ, μ̃ₖ₋₁) = sym_givens(μbarₖ₋₁, θ̃ₖ)
        ψ̃ₖ₋₁ =  s̃dₖ * η̃ₖ
        μ̃ₖ   = -c̃dₖ * η̃ₖ
        # Solve L̃ₖz̃ₖ = t̃ₖ where t̃ₖ = (ζ₁, …, ζₖ₋₁, ζbarₖ).
        # Only the last two components of z̃ₖ differ from the ones of the NELQ iterate.
        τ̃ₖ₋₁ = (ξₖ₋₁ - ψ̃ₖ₋₂ * τₖ₋₂) / μ̃ₖ₋₁
        τ̃ₖ   = (ζbarₖ - ρ̃ₖ₋₂ * τₖ₋₂ - ψ̃ₖ₋₁ * τ̃ₖ₋₁) / μ̃ₖ
        # ‖x⋆ - xᴺₖ‖² ≤ τ̃ₖ₋₁² + τ̃ₖ²
        err_nelq² = τ̃ₖ₋₁ * τ̃ₖ₋₁ + τ̃ₖ * τ̃ₖ
        # ‖x⋆ - xₖ‖² ≤ τ̃ₖ₋₁² + τ̃ₖ² - τₖ₋₁² - τₖ²
        err_qlp² = err_nelq² - τₖ₋₁ * τₖ₋₁ - τₖ * τₖ
        err_ubnd_qlp = sqrt(abs(err_qlp²))
        history && push!(err_ubnds_nelq, sqrt(err_nelq²))
        history && push!(err_ubnds_qlp, err_ubnd_qlp)
        # ‖xₖ‖² = ‖xᴺₖ‖² + τₖ₋₁² + τₖ²
        xNorm_qlp = sqrt(xNorm²_nelq + τₖ₋₁ * τₖ₋₁ + τₖ * τₖ)
        fwd_err_ubnd = err_ubnd_qlp ≤ etol * xNorm_qlp
      end
      xNorm = knorm(n, x)
      backward = rNorm / (ANorm * xNorm)

      # Update stopping criterion.
      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      ill_cond_mach = (one(T) + inv(Acond) ≤ one(T))
      zero_resid_mach = (one(T) + backward ≤ one(T))

      # Stopping conditions based on user-provided tolerances.
      tired = iter ≥ itmax
      resid_decrease_lim = (rNorm ≤ ε)
      zero_resid_lim = MisI && (backward ≤ eps(T))
      breakdown = βₖ₊₁ ≤ btol

      user_requested_exit = callback(workspace) :: Bool
      zero_resid = zero_resid_mach | zero_resid_lim
      resid_decrease = resid_decrease_lim
      solved = resid_decrease | zero_resid | fwd_err_ubnd
      inconsistent = (ArNorm ≤ κ && abs(μbarₖ) ≤ Artol) || (breakdown && !solved)
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns

      # Update variables
      if iter ≥ 2
        sₖ₋₂ = sₖ₋₁
        cₖ₋₂ = cₖ₋₁
        ξₖ₋₁ = ξₖ
        μbisₖ₋₂ = μbisₖ₋₁
        ψbarₖ₋₂ = ψbarₖ₋₁
      end
      if nelq_residuals
        if iter ≥ 2
          χoldₖ₋₂ = χₖ₋₁
          χoldₖ₋₁ = χₖ
          γₖ₋₂ = γₖ₋₁
        end
        νoldₖ₋₂ = νₖ₋₁
        νoldₖ₋₁ = νₖ
        κₖ₋₁ = κₖ
        λₖ₋₂ = λₖ₋₁
        λₖ₋₁ = λₖ
        αₖ₋₁ = αₖ
        βₖ₋₁ = βₖ
      end
      sₖ₋₁ = sₖ
      cₖ₋₁ = cₖ
      μbarₖ₋₁ = μbarₖ
      ζbarₖ = ζbarₖ₊₁
      βₖ = βₖ₊₁
      if nelq_residuals
        kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %9.1e  %7.1e  %7.1e  %8.1e  %7.1e  %7.1e  %8.1e  %.2fs\n", iter, rNorm, ArNorm, rNorm_nelq, ArNorm_nelq, βₖ₊₁, λₖ, μbarₖ, ANorm, Acond, backward, start_time |> ktimer)
      else
        kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %7.1e  %7.1e  %8.1e  %.2fs\n", iter, rNorm, ArNorm, βₖ₊₁, λₖ, μbarₖ, ANorm, Acond, backward, start_time |> ktimer)
      end
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Finalize the update of x
    if iter ≥ 2
      kaxpy!(n, τₖ₋₁, wₖ₋₁, x)
    end
    if !inconsistent
      kaxpy!(n, τₖ, wₖ, x)
    end

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    ill_cond_mach       && (status = "condition number seems too large for this machine")
    inconsistent        && (status = "found approximate minimum least-squares solution")
    zero_resid          && (status = "found approximate zero-residual solution")
    solved              && (status = "solution good enough given atol and rtol")
    fwd_err_ubnd        && (status = "forward error upper bound small enough")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

   # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end