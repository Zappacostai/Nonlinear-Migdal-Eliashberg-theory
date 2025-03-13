using LinearAlgebra, QuadGK

function ReΔ_f(α²F::F, u, ωc, ω₀, λ₀, λ₁, ϵ, ωcrit, T, maxiter, N, fineω, maxω, kₘ, Δₘ, kB, zmin, zmax) where {F<:Function}

    function n_B(x, T, δ, kB)
        return real(1/(exp((x+im*δ)/(kB*T))-1))
    end
        
    function n_F(x, T, δ, kB)
        return real(1/(exp((x+im*δ)/(kB*T))+1))
    end

    function Integrand(α²F::F, x, ω₀, λ₀, λ₁, ϵ, ωcrit, ω, zmin, zmax) where {F<:Function}
        return 2α²F(x, ω₀, λ₀, λ₁, ϵ, ωcrit, zmin, zmax).*(x./(x.^2 .- ω.^2))
    end

    function Integrand!(y, α²F::F, x, ω₀, λ₀, λ₁, ϵ, ωcrit, ω, zmin, zmax) where {F<:Function}
        return y .= 2α²F.(x, ω₀, λ₀, λ₁, ϵ, ωcrit, zmin, zmax).*(x./(x.^2 .- ω.^2))
    end

    function λcalc!(λ, α²F::F, ω₀, λ₀, λ₁, ϵ, ωcrit, ω, buf, zmin, zmax) where {F<:Function}
        return quadgk!((y, x)-> Integrand!(y, α²F, x, ω₀, λ₀, λ₁, ϵ, ωcrit, ω, zmin, zmax), λ, zmin, zmax, rtol=1e-3, segbuf = buf)[1]
    end

    function self_consistent(pre,num,den,branch)
        return pre*num/den*sign(branch)
    end

    P = Int64(round(N/100))

    ω1 = LinRange(-fineω, fineω, 2(N-P));      ω2 = LinRange(fineω, maxω, P) .+ abs(ω1[1]-ω1[2]);         M = size(kₘ,1)
    ω = [-ω2; ω1; ω2]
    dω = [abs(ω[1] - ω[2]); abs.(ω[1:end-1] .- ω[2:end])]

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

    Z = (1+im)*(1 .+ Den);     W = (1+im)*Num

    SR = zeros(ComplexF64,2N)
    sumW1 = zeros(ComplexF64,2N);       sumZ1 = zeros(ComplexF64,2N)
    sumW2 = zeros(ComplexF64,2N);       sumZ2 = zeros(ComplexF64,2N)

    n1 = zeros(2N);    n2 = zeros(2N);  δ = 1e-10

    for l ∈ 1:maxiter
    
        @inline for n ∈ -N:N-1  #ω

            SR .= sqrt.(Complex.(((ω.+im*δ) .* Z).^2 .- W.^2))
            nᵢ = n+N+1;     Lᵢ = N-n

            @views n1[1:Lᵢ] .= α²F(ω[nᵢ:end] .- ω[nᵢ], ω₀, λ₀, λ₁, ϵ, ωcrit, zmin, zmax) .* (n_B.(ω[nᵢ:end] .- ω[nᵢ], T, δ, kB) .+ n_F.(ω[nᵢ:end], T, δ, kB))

            @views n2[1:nᵢ] .= α²F(-ω[nᵢ:-1:1] .+ ω[nᵢ], ω₀, λ₀, λ₁, ϵ, ωcrit, zmin, zmax) .* (n_B.(-ω[nᵢ:-1:1] .+ ω[nᵢ], T, δ, kB) .+ n_F.(-ω[nᵢ:-1:1], T, δ, kB))
            
            #∫dz 
            @views sumW1[1:Lᵢ] .= self_consistent.(n1[1:Lᵢ], W[nᵢ:end], SR[nᵢ:end], imag.(SR[nᵢ:end]))
            @views sumZ1[1:Lᵢ] .= self_consistent.(n1[1:Lᵢ], (ω[nᵢ:end].+im*δ).*Z[nᵢ:end], SR[nᵢ:end], imag.(SR[nᵢ:end]))
            #∫(-dz) 
            @views sumW2[1:nᵢ] .= self_consistent.(n2[1:nᵢ], W[nᵢ:-1:1], SR[nᵢ:-1:1], imag.(SR[nᵢ:-1:1]))
            @views sumZ2[1:nᵢ] .= self_consistent.(n2[1:nᵢ], (ω[nᵢ:-1:1].+im*δ).*Z[nᵢ:-1:1], SR[nᵢ:-1:1], imag.(SR[nᵢ:-1:1]))

            IntW1 = sum(sumW1)
            IntZ1 = sum(sumZ1)

            IntW2 = sum(sumW2)
            IntZ2 = sum(sumZ2)

            Z[nᵢ] = 1 + Den[nᵢ] + (im*π*dω[nᵢ]/ω[nᵢ]) * (IntZ1 + IntZ2)
            W[nᵢ] = Num[nᵢ] + im*π*dω[nᵢ]* (IntW1 + IntW2)

        end

    end

    Δ = W ./ Z

    return [Δ W Z ω]

end