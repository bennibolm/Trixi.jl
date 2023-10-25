# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function check_bounds(u, mesh::AbstractMesh{2}, equations, solver, cache,
                              limiter::SubcellLimiterIDP,
                              time, iter, output_directory, save_errors)
    (; local_minmax, positivity, spec_entropy, math_entropy) = solver.volume_integral.limiter
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    (; idp_bounds_delta) = limiter.cache

    if local_minmax
        for v in limiter.local_minmax_variables_cons
            v_string = string(v)
            key_min = Symbol(v_string, "_min")
            key_max = Symbol(v_string, "_max")
            deviation_min = idp_bounds_delta[key_min]
            deviation_max = idp_bounds_delta[key_max]
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                var = u[v, i, j, element]
                deviation_min[1] = max(deviation_min[1],
                                       variable_bounds[key_min][i, j, element] - var)
                deviation_max[1] = max(deviation_max[1],
                                       var - variable_bounds[key_max][i, j, element])
            end
            deviation_min[2] = max(deviation_min[2], deviation_min[1])
            deviation_max[2] = max(deviation_max[2], deviation_max[1])
        end
    end
    if spec_entropy
        key = :spec_entropy_min
        deviation = idp_bounds_delta[key]
        for element in eachelement(solver, cache), j in eachnode(solver),
            i in eachnode(solver)

            s = entropy_spec(get_node_vars(u, equations, solver, i, j, element),
                             equations)
            deviation[1] = max(deviation[1], variable_bounds[key][i, j, element] - s)
        end
        deviation[2] = max(deviation[2], deviation[1])
    end
    if math_entropy
        key = :math_entropy_max
        deviation = idp_bounds_delta[key]
        for element in eachelement(solver, cache), j in eachnode(solver),
            i in eachnode(solver)

            s = entropy_math(get_node_vars(u, equations, solver, i, j, element),
                             equations)
            deviation[1] = max(deviation[1], s - variable_bounds[key][i, j, element])
        end
        deviation[2] = max(deviation[2], deviation[1])
    end
    if positivity
        for v in limiter.positivity_variables_cons
            key = Symbol(string(v), "_min")
            deviation = idp_bounds_delta[key]
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                var = u[v, i, j, element]
                deviation[1] = max(deviation[1],
                                   variable_bounds[key][i, j, element] - var)
            end
            deviation[2] = max(deviation[2], deviation[1])
        end
        for variable in limiter.positivity_variables_nonlinear
            key = Symbol(string(variable), "_min")
            deviation = idp_bounds_delta[key]
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                var = variable(get_node_vars(u, equations, solver, i, j, element),
                               equations)
                deviation[1] = max(deviation[1],
                                   variable_bounds[key][i, j, element] - var)
            end
            deviation[2] = max(deviation[2], deviation[1])
        end
    end
    if save_errors
        # Print to output file
        open("$output_directory/deviations.txt", "a") do f
            print(f, iter, ", ", time)
            if local_minmax
                for v in limiter.local_minmax_variables_cons
                    v_string = string(v)
                    print(f, ", ", idp_bounds_delta[Symbol(v_string, "_min")][1],
                          idp_bounds_delta[Symbol(v_string, "_max")][1])
                end
            end
            if spec_entropy
                print(f, ", ", idp_bounds_delta[:spec_entropy_min][1])
            end
            if math_entropy
                print(f, ", ", idp_bounds_delta[:math_entropy_max][1])
            end
            if positivity
                for v in limiter.positivity_variables_cons
                    print(f, ", ", idp_bounds_delta[Symbol(string(v), "_min")][1])
                end
                for variable in limiter.positivity_variables_nonlinear
                    print(f, ", ",
                          idp_bounds_delta[Symbol(string(variable), "_min")][1])
                end
            end
            println(f)
        end
        # Reset first entries of idp_bounds_delta
        for (key, _) in idp_bounds_delta
            idp_bounds_delta[key][1] = zero(eltype(idp_bounds_delta[key][1]))
        end
    end

    return nothing
end

@inline function check_bounds(u, mesh::AbstractMesh{2}, equations, solver, cache,
                              limiter::SubcellLimiterMCL,
                              time, iter, output_directory, save_errors)
    (; var_min, var_max) = limiter.cache.subcell_limiter_coefficients
    (; bar_states1, bar_states2, lambda1, lambda2) = limiter.cache.container_bar_states
    (; idp_bounds_delta) = limiter.cache
    (; antidiffusive_flux1, antidiffusive_flux2) = cache.antidiffusive_fluxes

    n_vars = nvariables(equations)

    if limiter.DensityLimiter
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1],
                                                var_min[1, i, j, element] -
                                                u[1, i, j, element])
                idp_bounds_delta[1, 2, 1] = max(idp_bounds_delta[1, 2, 1],
                                                u[1, i, j, element] -
                                                var_max[1, i, j, element])
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        # Checking the bounds for...
        # - density (rho):
        #   \bar{rho}^{min} <= \bar{rho}^{Lim} <= \bar{rho}^{max}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1],
                                                var_min[1, i, j, element] - rho_limited)
                idp_bounds_delta[1, 2, 1] = max(idp_bounds_delta[1, 2, 1],
                                                rho_limited - var_max[1, i, j, element])
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1],
                                                var_min[1, i, j, element] - rho_limited)
                idp_bounds_delta[1, 2, 1] = max(idp_bounds_delta[1, 2, 1],
                                                rho_limited - var_max[1, i, j, element])
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1],
                                                var_min[1, i, j, element] - rho_limited)
                idp_bounds_delta[1, 2, 1] = max(idp_bounds_delta[1, 2, 1],
                                                rho_limited - var_max[1, i, j, element])
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1],
                                                var_min[1, i, j, element] - rho_limited)
                idp_bounds_delta[1, 2, 1] = max(idp_bounds_delta[1, 2, 1],
                                                rho_limited - var_max[1, i, j, element])
            end
        end
    end # limiter.DensityLimiter

    if limiter.SequentialLimiter
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                for v in 2:n_vars
                    var_limited = u[v, i, j, element] / u[1, i, j, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited -
                                                    var_max[v, i, j, element])
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure = 0.5 *
                                     (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                     u[1, i, j, element] * u[4, i, j, element]
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                end
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        # Checking the bounds for...
        # - velocities and energy (phi):
        #   \bar{phi}^{min} <= \bar{phi}^{Lim} / \bar{rho}^{Lim} <= \bar{phi}^{max}
        # - pressure (p):
        #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2
        var_limited = zero(eltype(idp_bounds_delta))
        error_pressure = zero(eltype(idp_bounds_delta))
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i, j, element] -
                                  antidiffusive_flux1[v, i, j, element] /
                                  lambda1[i, j, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited / rho_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited / rho_limited -
                                                    var_max[v, i, j, element])
                    if limiter.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i + 1, j, element] +
                                  antidiffusive_flux1[v, i + 1, j, element] /
                                  lambda1[i + 1, j, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited / rho_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited / rho_limited -
                                                    var_max[v, i, j, element])
                    if limiter.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j, element] -
                                  antidiffusive_flux2[v, i, j, element] /
                                  lambda2[i, j, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited / rho_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited / rho_limited -
                                                    var_max[v, i, j, element])
                    if limiter.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j + 1, element] +
                                  antidiffusive_flux2[v, i, j + 1, element] /
                                  lambda2[i, j + 1, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited / rho_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited / rho_limited -
                                                    var_max[v, i, j, element])
                    if limiter.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
            end
        end
    elseif limiter.ConservativeLimiter
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                for v in 2:n_vars
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    u[v, i, j, element])
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    u[v, i, j, element] -
                                                    var_max[v, i, j, element])
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure = 0.5 *
                                     (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                     u[1, i, j, element] * u[4, i, j, element]
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                end
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        # Checking the bounds for...
        # - conservative variables (phi):
        #   \bar{rho*phi}^{min} <= \bar{rho*phi}^{Lim} <= \bar{rho*phi}^{max}
        # - pressure (p):
        #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2
        var_limited = zero(eltype(idp_bounds_delta))
        error_pressure = zero(eltype(idp_bounds_delta))
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i, j, element] -
                                  antidiffusive_flux1[v, i, j, element] /
                                  lambda1[i, j, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited -
                                                    var_max[v, i, j, element])
                    if limiter.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i + 1, j, element] +
                                  antidiffusive_flux1[v, i + 1, j, element] /
                                  lambda1[i + 1, j, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited -
                                                    var_max[v, i, j, element])
                    if limiter.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j, element] -
                                  antidiffusive_flux2[v, i, j, element] /
                                  lambda2[i, j, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited -
                                                    var_max[v, i, j, element])
                    if limiter.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j + 1, element] +
                                  antidiffusive_flux2[v, i, j + 1, element] /
                                  lambda2[i, j + 1, element]
                    idp_bounds_delta[1, 1, v] = max(idp_bounds_delta[1, 1, v],
                                                    var_min[v, i, j, element] -
                                                    var_limited)
                    idp_bounds_delta[1, 2, v] = max(idp_bounds_delta[1, 2, v],
                                                    var_limited -
                                                    var_max[v, i, j, element])
                    if limiter.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                              n_vars + 1],
                                                             error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
            end
        end
    elseif limiter.PressurePositivityLimiterKuzmin
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                error_pressure = 0.5 * (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                 u[1, i, j, element] * u[4, i, j, element]
                idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                          n_vars + 1],
                                                         error_pressure)
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        # Checking the bounds for...
        # - pressure (p):
        #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                error_pressure = 0.5 *
                                 (bar_states1[2, i, j, element] -
                                  antidiffusive_flux1[2, i, j, element] /
                                  lambda1[i, j, element])^2 +
                                 0.5 *
                                 (bar_states1[3, i, j, element] -
                                  antidiffusive_flux1[3, i, j, element] /
                                  lambda1[i, j, element])^2 -
                                 (bar_states1[4, i, j, element] -
                                  antidiffusive_flux1[4, i, j, element] /
                                  lambda1[i, j, element]) * rho_limited
                idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                          n_vars + 1],
                                                         error_pressure)
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                error_pressure = 0.5 *
                                 (bar_states1[2, i + 1, j, element] +
                                  antidiffusive_flux1[2, i + 1, j, element] /
                                  lambda1[i + 1, j, element])^2 +
                                 0.5 *
                                 (bar_states1[3, i + 1, j, element] +
                                  antidiffusive_flux1[3, i + 1, j, element] /
                                  lambda1[i + 1, j, element])^2 -
                                 (bar_states1[4, i + 1, j, element] +
                                  antidiffusive_flux1[4, i + 1, j, element] /
                                  lambda1[i + 1, j, element]) * rho_limited
                idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                          n_vars + 1],
                                                         error_pressure)
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                error_pressure = 0.5 *
                                 (bar_states2[2, i, j, element] -
                                  antidiffusive_flux2[2, i, j, element] /
                                  lambda2[i, j, element])^2 +
                                 0.5 *
                                 (bar_states2[3, i, j, element] -
                                  antidiffusive_flux2[3, i, j, element] /
                                  lambda2[i, j, element])^2 -
                                 (bar_states2[4, i, j, element] -
                                  antidiffusive_flux2[4, i, j, element] /
                                  lambda2[i, j, element]) * rho_limited
                idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                          n_vars + 1],
                                                         error_pressure)
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                error_pressure = 0.5 *
                                 (bar_states2[2, i, j + 1, element] +
                                  antidiffusive_flux2[2, i, j + 1, element] /
                                  lambda2[i, j + 1, element])^2 +
                                 0.5 *
                                 (bar_states2[3, i, j + 1, element] +
                                  antidiffusive_flux2[3, i, j + 1, element] /
                                  lambda2[i, j + 1, element])^2 -
                                 (bar_states2[4, i, j + 1, element] +
                                  antidiffusive_flux2[4, i, j + 1, element] /
                                  lambda2[i, j + 1, element]) * rho_limited
                idp_bounds_delta[1, 1, n_vars + 1] = max(idp_bounds_delta[1, 1,
                                                                          n_vars + 1],
                                                         error_pressure)
            end
        end
    end # limiter.PressurePositivityLimiterKuzmin

    if limiter.DensityPositivityLimiter
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1],
                                                -u[1, i, j, element])
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        beta = limiter.DensityPositivityCorrectionFactor
        # Checking the bounds for...
        # - density (rho):
        #   beta * \bar{rho} <= \bar{rho}^{Lim}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = (1 - beta) * bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1], -rho_limited)
                # +x
                rho_limited = (1 - beta) * bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1], -rho_limited)
                # -y
                rho_limited = (1 - beta) * bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1], -rho_limited)
                # +y
                rho_limited = (1 - beta) * bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                idp_bounds_delta[1, 1, 1] = max(idp_bounds_delta[1, 1, 1], -rho_limited)
            end
        end
    end # limiter.DensityPositivityLimiter

    for v in eachvariable(equations)
        idp_bounds_delta[2, 1, v] = max(idp_bounds_delta[2, 1, v],
                                        idp_bounds_delta[1, 1, v])
        idp_bounds_delta[2, 2, v] = max(idp_bounds_delta[2, 2, v],
                                        idp_bounds_delta[1, 2, v])
    end
    if limiter.PressurePositivityLimiterKuzmin
        idp_bounds_delta[2, 1, n_vars + 1] = max(idp_bounds_delta[2, 1, n_vars + 1],
                                                 idp_bounds_delta[1, 1, n_vars + 1])
    end

    if !save_errors
        return nothing
    end
    open("$output_directory/deviations.txt", "a") do f
        print(f, iter, ", ", time)
        for v in eachvariable(equations)
            print(f, ", ", idp_bounds_delta[1, 1, v], ", ", idp_bounds_delta[1, 2, v])
        end
        if limiter.PressurePositivityLimiterKuzmin
            print(f, ", ", idp_bounds_delta[1, 1, n_vars + 1])
        end
        println(f)
    end
    for v in eachvariable(equations)
        idp_bounds_delta[1, 1, v] = zero(eltype(idp_bounds_delta))
        idp_bounds_delta[1, 2, v] = zero(eltype(idp_bounds_delta))
    end
    if limiter.PressurePositivityLimiterKuzmin
        idp_bounds_delta[1, 1, n_vars + 1] = zero(eltype(idp_bounds_delta))
    end

    return nothing
end
end # @muladd
