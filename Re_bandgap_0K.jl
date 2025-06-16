using LinearAlgebra, QuadGK

function ReΔ_0(α²F::F, u, ωc, ω₀, λ₀, λ₁, ϵ, ωcrit, N, fineω, maxω, kₘ, Δₘ, kB, zmin, zmax, T) where {F<:Function}

    function Integrand!(y, α²F::F, x, ω₀, λ₀, λ₁, ϵ, ωcrit, ω, zmin, zmax) where {F<:Function}
        return y .= 2α²F.(x, ω₀, λ₀, λ₁, ϵ, ωcrit, zmin, zmax).*(x./(x.^2 .- ω.^2))
    end

    function λcalc!(λ, α²F::F, ω₀, λ₀, λ₁, ϵ, ωcrit, ω, buf, zmin, zmax) where {F<:Function}
        return quadgk!((y, x)-> Integrand!(y, α²F, x, ω₀, λ₀, λ₁, ϵ, ωcrit, ω, zmin, zmax), λ, zmin, zmax, rtol=1e-3, segbuf = buf)[1]
    end

    P = Int64(round(N/100))

    ω1 = LinRange(-fineω, fineω, 2(N-P));      ω2 = LinRange(fineω, maxω, P) .+ abs(ω1[1]-ω1[2]);         M = size(kₘ,1)
    ω = [-reverse(ω2); ω1; ω2]

    Num = zeros(ComplexF64,2N);         Den = zeros(ComplexF64,2N);     
    λ = zeros(ComplexF64,M);            sumNum = zeros(ComplexF64,M);       sumDen = zeros(ComplexF64,M);

    buf = alloc_segbuf(Float64, Vector{ComplexF64}, Float64)

    @inline for n ∈ 1:2N
        
        λcalc!(λ, α²F, ω₀, λ₀, λ₁, ϵ, ωcrit, ω[n] .- im.*kₘ, buf, zmin, zmax)

        @. sumNum = Δₘ / sqrt(kₘ^2 + Δₘ^2) * (λ - u*(abs(kₘ < ωc)))
        @. sumDen = kₘ / sqrt(kₘ^2 + Δₘ^2) * λ

        Num[n] = π*kB*T*sum(sumNum)
        Den[n] = (im*π*kB*T/ω[n])*sum(sumDen)

    end

    Z = (1 .+ Den);     W = Num

    Δ = W ./ Z

    return [Δ W Z ω]

end