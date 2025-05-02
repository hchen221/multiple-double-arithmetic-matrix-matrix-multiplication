"""
TCCuda.jl uses CUDA to compute matrix matrix multiplication in double arithmetic
"""

include("TCflat.jl")

using CUDA

"""
matmul!(A,B,C) is a dot product kernel that accumulates the results of A*B to C where A,B,C are nxn matrices of doubles
"""
function matmul!(A,B,C)
    n = gridDim().x*blockDim().x
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    j = (blockIdx().y-1)*blockDim().y+threadIdx().y
    for k=1:n
        CUDA.@atomic C[(i-1)*n+j] += A[(i-1)*n+k]*B[(k-1)*n+j]
    end
    return nothing
end

"""                                                                                                                    
dotconv!(A,B,C) is a dot product kernel that accumulates the results of A*B to C where A,B,C are nxn matrices of p-doubles. It assumes nxn blocks and pxp threads per block
"""
function dotconv!(A,B,C)
    n,p = gridDim().x,blockDim().x
    I,J = blockIdx().x,blockIdx().y
    i,j = threadIdx().x,threadIdx().y
    if i < j
        for k=1:n
            CUDA.@atomic C[(I-1)*n*p+(J-1)*p+i] += 0
        end
    else
        #@cuprintln("Computing part ($i,$(j+1-i)) of C[$I,$J]($j)\n")
        for k=1:n
            CUDA.@atomic C[(I-1)*n*p+(J-1)*p+i] += A[(I-1)*n*p+(k-1)*p+j]*B[(k-1)*n*p+(J-1)*p+(i+1-j)]
        end
    end
    return nothing
end
# try using negative indices and each term in the convolution does p computations to avoid thread divergence
# Refer to convolution formula in https://homepages.math.uic.edu/~jan/convolutions.pdf

"""
dotconvbutbetter!(A,B,C) runs nxn blocks of px1 threads to do the same as dotconv!(A,B,C), doesn't rely on atomic add
"""
function dotconvbutbetter!(A,B,C)
    n,p = gridDim().x,blockDim().x
    I,J = blockIdx().x,blockIdx().y
    i = threadIdx().x
    for k=1:n
        for j=1:p
            if j+i-p>0
                a = A[(I-1)*n*p+(k-1)*p+(j+i-p)]
            else
                a = 0
            end
            b = B[(k-1)*n*p+(J-1)*p+(p+1-j)]
            C[(I-1)*n*p+(J-1)*p+i] += a*b
        end
    end
    return nothing
end

"""
matconv!(A,B,C,n,p) computes A*B and adds it to C, where A,B,C are nxn matrices of p-doubles
"""
function matconv!(A,B,C,n,p)
    function matconvhelp1(i,j)
        Cbuffer = CUDA.zeros(Float64,n^2)
        @cuda threads=(n,n) blocks=(1,1) matmul!(A[i:p:end],B[j:p:end],Cbuffer)
        return Cbuffer
    end
    function matconvhelp2(k)
        C[k:p:end] += sum(matconvhelp1.(1:k,k:-1:1))
        return nothing
    end
    matconvhelp2.(1:p)
    return nothing
end
