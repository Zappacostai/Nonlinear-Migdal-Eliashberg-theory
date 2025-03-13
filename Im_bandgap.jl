using LinearAlgebra, QuadGK

function ImΔₘ(α²F::F, u, ωc, ω₀, λ₀, λ₁, ϵ, ωcrit, T, maxiter, zmin, zmax) where {F<:Function}

    function λcalc(α²F::F, ω₀, λ₀, λ₁, ϵ, ωcrit, ωₘ, zmin, zmax) where {F<:Function}
        return quadgk(x-> 2α²F(x,ω₀,λ₀,λ₁,ϵ,ωcrit, zmin, zmax)*(x./(x^2 .+ ωₘ.^2)), zmin, zmax, rtol=1e-4)[1]
    end

    kB = 8.617333e-5
    N = Int64(round.(ωc./(2π*kB*T).-1/2))

    m = -N:N-1

        kₘ = (2m.+1)π*kB*T
        λ = λcalc(α²F, ω₀, λ₀, λ₁, ϵ, ωcrit, (2(m.+N))π*kB*T, zmin, zmax)
    
    Δ = -ones(2N)

    for l ∈ 1:maxiter
    
        numiSum = zeros(2N)
        deniSum = zeros(2N)

        @inline for n ∈ -N:N-1

            @views numiSum .= π*kB*T .* Δ ./ sqrt.( kₘ.^2 .+ Δ.^2 ) .* (λ[abs.(m.-n).+1] .- u*(abs.(kₘ) .< ωc))
            @views deniSum .= (1/(2n+1) .* kₘ .* λ[abs.(m.-n).+1]) ./ sqrt.(kₘ.^2 .+ Δ.^2 )

            Numᵢ = sum(numiSum)
            Denᵢ = sum(deniSum)

            Δ[n+N+1] = Numᵢ / (1 + Denᵢ) 
    
        end

    end

    Δ .= Δ.*sign(-Δ[end]) #we choose the solution with negative asymptote

    return [Δ kₘ]

end