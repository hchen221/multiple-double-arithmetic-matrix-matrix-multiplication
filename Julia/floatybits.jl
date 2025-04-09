"""
floatybits.jl contains all functions used to generate random double doubles, big floats, and work between bit representation and numerical representation
"""

"""
val_bits(bits) parses an array of bits and returns the numerical value
"""
function val_bits(bits)
    s = 0
    for i=1:size(bits)[1]
        s = 2*s+bits[i]
    end
    return s
end

"""
expandfracbits(fr) takes fr in [0,1] on input and returns a 52 bit representation 
"""
function expandfracbits(fr)
    bits = []
    temp = fr
    i = -1
    while temp>0
        if temp >= 2.0^i
            push!(bits,1)
            temp -= 2.0^i
        else
            push!(bits,0)
        end
        i -= 1
    end
    while size(bits)[1] < 52
        push!(bits,0)
    end
    return bits
end

"""
bigfracbits(FR) expands the bits for the fraction of a big float (specifically the sum of a double double) and fills it in 104 bits
"""
function bigfracbits(FR,p)
    bits = expandfracbits(FR)
    while size(bits)[1] < 52*p
        push!(bits,0)
    end
    return bits
end

"""
expandexpbits(n) takes an exponent integer n on input and returns an 11 bit representation
"""
function expandexpbits(n)
    bits = []
    temp = n
    while temp>0
        pushfirst!(bits,temp%2)
        temp = Int(floor(temp/2))
    end
    while size(bits)[1] < 11
        pushfirst!(bits,0)
    end
    return bits
end

"""
bitform(x) take a double x on input and returns the 64 bit representation
"""
function bitform(x)
    fraction,exponent = frexp(x)
    return vcat(vcat([Int(x<0)],expandexpbits(exponent+1023)),expandfracbits(fraction))
end

"""
random_double_bits(signbit,exponent) returns the bit representation of a random double with a given sign bit and exponent
"""
random_double_bits(signbit,exponent) = vcat(vcat([signbit],expandexpbits(exponent)),Int64.(round.(rand(52))))

"""
double_rep(bits) returns the numerical double of a 64 bit number
"""
double_rep(bits) = (-1)^bits[1]*2.0^(val_bits(bits[2:12])-1023)*(sum([bits[12+i]*2.0^(-i) for i=1:52]))

"""
random_pd(expmin,expmax) returns a random p-double
"""
function random_pd(expmin,expmax,p=2)
    hiexp = rand(expmin+1023:expmax+1023)
    parts = [random_double_bits(0,hiexp-52*(i-1)) for i=1:p]
    return double_rep.(parts)
end

"""
split4(bits) takes a 64 bit representation on input and returns the quad double in numerical form
"""
function split4(bits)
    bits1 = deepcopy(bits)
    bits1[26:end] .= 0
    bits2 = deepcopy(bits)
    bits2[13:25] .= 0
    bits2[39:end] .= 0
    bits3 = deepcopy(bits)
    bits3[13:38] .= 0
    bits3[52:end] .= 0
    bits4 = deepcopy(bits)
    bits4[13:51] .= 0
    D = double_rep.([bits1,bits2,bits3])
    push!(D,double_rep(bits)-sum(D))
    return D
end

"""
split4pd(x) applies split4 to a p-double x then returns the combined 4p-double
"""
split4pd(x) = mapreduce(permutedims,hcat,split4.(bitform.(x)))[1,:]

"""
Big(x) converts x component wise to BigFloat, used for vectorization purposes
"""
Big(x) = BigFloat.(x)

"""
BFtoPD(X) takes a big float X on input (that's the sum of a p-double) and returns it to p-double format
"""
function BFtoPD(X,p)
    F,E = frexp(X)
    Fbits = bigfracbits(F,p)
    return double_rep.([vcat(vcat([0],expandexpbits(E+1023-52*(i-1))),Fbits[52*(i-1)+1:52*i]) for i=1:p])
end

"""
vecpart(x,i) returns the ith component of x, used for vectorization purposes
"""
vecpart(x,i) = x[i]
