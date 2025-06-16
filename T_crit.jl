using LinearAlgebra, QuadGK

function T_crit(α²F::F, u, ωc, dT, ω₀, λ₀, λ₁, ϵ, ωcrit, zmin, zmax, minT) where {F<:Function}

    function λcalc(α²F::F, ω₀, λ₀, λ₁, ϵ, ωcrit, ωₘ, zmin, zmax) where {F<:Function}
        return quadgk(x-> 2α²F(x,ω₀,λ₀,λ₁,ϵ,ωcrit,zmin,zmax).*(x./(x^2 .+ ωₘ.^2)), zmin, zmax, rtol=1e-5)[1] 
    end

    kB = 8.617333e-5 #eV/K
    maxT = 10000

    λ0 = λ₀

    j = 0;      T = maxT;     C = -1

    while abs(T - C) > dT

        j += 1

        N = Int64(round(minimum([ωc/(2π*kB*T)-1/2 ωc/(2π*kB*minT)-1/2])))
        Z = zeros(2N)
        M = zeros(2N,2N)

        n = -N:N-1;     ω = (2n.+1)π*kB*T

        qₙ = 2(n.+N)π*kB*T

        λ1 = λ₁*(1 + 2 / (exp(ω₀/(kB*T)) - 1))

        λ = λcalc(α²F, ω₀, λ0, λ1, ϵ, ωcrit, qₙ, zmin, zmax)

        @inline for m ∈ -N:N-1

            nmax = Int64(floor(abs(m+1/2))+1);      
            @views Λ = sum(λ[2:nmax])
            
            if m==0 && m==-1
                Λ = 0
            end
    
            Z[m+N+1] = 1 + (π*kB*T/abs(ω[m+N+1]))*(λ[1] + 2Λ)
    
            @views M[m+N+1,n.+N.+1] .= Z[m+N+1].*I[m+N+1,n.+N.+1] .- (π*kB*T./abs.(ω[n.+N.+1])).*(λ[abs.(n.-m).+1] .- u*(abs.(ω[n.+N.+1]) .< ωc))
            
        end

        D = det(M);     C = T

        maxT = maxT/2

        if D > 0
            T -= maxT
        else
            T += maxT
            if j == 2
                println("Error, initial value.")
                break
            end
        end

        if j > 100
            println("Error, no convergence.")
            return j
            break
        end

    end

    return T

end