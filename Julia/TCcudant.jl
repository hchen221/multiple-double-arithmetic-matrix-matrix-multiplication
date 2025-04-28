"""
TCCudant.jl simulates CUDA to compute matrix matrix multiplication in double arithmetic without actually using CUDA but vectorization instead, for test purposes
"""

include("TCflat.jl")

"""
matmul!(A,B,C,n,i,j) is a dot product kernel that accumulates the results of A*B to C where A,B,C are nxn matrices of doubles
Computes C[i,j]
"""
function matmul!(A,B,C,n,i,j)
    for k=1:n
        C[(i-1)*n+j] += A[(i-1)*n+k]*B[(k-1)*n+j]
    end
    return nothing
end

"""
dotconv!(A,B,C) is a dot product kernel that accumulates the results of A*B to C where A,B,C are nxn matrices of p-doubles. It assumes nxn blocks and pxp threads per block
Computes the jth inner product as part of the convolution for the ith part of C[I,J]
"""
function dotconv!(A,B,C,n,p,I,J,i,j)
    res = 0
    if i < j
        for k=1:n
            C[(I-1)*n*p+(J-1)*p+i] += 0
        end
    else
        for k=1:n
            C[(I-1)*n*p+(J-1)*p+i] += A[(I-1)*n*p+(k-1)*p+j]*B[(k-1)*n*p+(J-1)*p+(i+1-j)]
        end
    end
    return nothing
end

"""
matconv!(A,B,C,n,p) computes A*B and adds it to C, where A,B,C are nxn matrices of p-doubles
"""
function matconv!(A,B,C,n,p)
    function matconvhelp1(i,j)
        Cbuffer = zeros(n^2)
        I,J = meshgrid(n)
        matmul!.(Ref(A[i:p:end]),Ref(B[j:p:end]),Ref(Cbuffer),Ref(n),I,J)
        return Cbuffer
    end
    function matconvhelp2(k)
        C[k:p:end] += sum(matconvhelp1.(1:k,k:-1:1))
        return nothing
    end
    matconvhelp2.(1:p)
    return nothing
end
