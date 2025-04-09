"""
TCflat.jl contains functions and helper functions including dot product computations and convolutions used to simulate a TensorCore application for matrix matrix multiplication
It specifically works with matrices of p-doubles as flattened arrays
"""

using LinearAlgebra
include("floatybits.jl")

"""
CUDA no like matrices. So flat(A) takes a matrix of p doubles A and flattens it into a vector
The vector strings the rows together, each row stringing the p doubles together
"""
flat(A) = vec(mapreduce(permutedims,vcat,A'))

"""
Given an nxn matrix A of p doubles flattened
A[ent(n,p,i,j)] returns A[i,j]
A[row(n,p,i)] returns A[i,:]
A[rowslice(n,p,i,j1,j2)] returns A[i,j1:j2]
A[col(n,p,j)] returns A[:,j]
A[frag(n,p,i1,i2,j1,j2)] returns A[i1:i2,j1:j2]
All still in flattened form
The jth part of the p double component wise could be taken simply by x[j:p:end]
"""
ent(n,p,i,j) = (i-1)*n*p+(j-1)*p+1:(i-1)*n*p+j*p
row(n,p,i) = (i-1)*n*p+1:i*n*p
rowslice(n,p,i,j1,j2) = row(n,p,i)[(j1-1)*p+1:j2*p]
col(n,p,j) = vec(mapreduce(permutedims,hcat,ent.(Ref(n),Ref(p),1:n,Ref(j))))
frag(n,p,i1,i2,j1,j2) = vec(mapreduce(permutedims,hcat,rowslice.(Ref(n),Ref(p),i1:i2,Ref(j1),Ref(j2))))

"""
flatmyconv(x,y,p,f) computes f(x,y) using convolutions. flatconvpart and flatSconv are helper functions
"""
flatconvpart(x,y,p,i,j,f) = f(x[i:p:end],y[j:p:end]) # apply the function to the ith part of x and the jth part of y
flatSconv(x,y,p,i,f) = sum(flatconvpart.(Ref(x),Ref(y),Ref(p),1:i,i:-1:1,Ref(f))) # Compute the ith term in the convolution
flatmyconv(x,y,p,f) = flatSconv.(Ref(x),Ref(y),Ref(p),1:p,Ref(f)) # Sum up all terms in the convolution

"""
meshgrid(n) essentially returns numpy.meshgrid(1:n,1:n)
"""
function meshgrid(n)
    I = ones(Int,n,n)
    J = ones(Int,n,n)
    function ind_assign(i)
        I[i,:] .= i
        J[:,i] .= i
    end
    ind_assign.(2:n)
    return I,J
end

"""
zp(p) returns 0 as a p double
"""
zp(p) = zeros(p)
"""
TCKernel!(A,B,C,n,p) is used to simulate a TensorCore kernel performing A*B where A,B are nxn with p doubles and filling it to C
dotapply! is a helper function
"""
function flatdotapply!(A,B,C,n,p,i,j)
    C[ent(n,p,i,j)] += flatmyconv(A[row(n,p,i)],B[col(n,p,j)],p,dot)
    return nothing
end
function flatTCKernel!(A,B,C,n,p)
    I,J = meshgrid(n)
    flatdotapply!.(Ref(A),Ref(B),Ref(C),Ref(n),Ref(p),I,J)
    return nothing
end

"""
flatTCfrag(A,B,C,n,p,nfrag,ia,ib,ja,jb) computes the matrix product of the (ia,ja) tile of A and the (ib,jb) tile of B
then adds the result to C[ia:ib,ja:jb]
"""
function flatTCfrag!(A,B,C,n,p,nfrag,ia,ib,ja,jb)
    Cfrag = zeros(nfrag^2*p)
    flatTCKernel!(
        A[frag(n,p,(ia-1)*nfrag+1,ia*nfrag,(ja-1)*nfrag+1,ja*nfrag)],
        B[frag(n,p,(ib-1)*nfrag+1,ib*nfrag,(jb-1)*nfrag+1,jb*nfrag)],
        Cfrag,nfrag,p)
    C[frag(n,p,(ia-1)*nfrag+1,ia*nfrag,(jb-1)*nfrag+1,jb*nfrag)] += Cfrag
    return nothing
end
"""
fraginnerproduct(A,B,i,j,nfrag,nlen) interprets A,B as nlen x nlen matrices where each entry is an nfrag x nfrag matrix
It computes the inner product of the ith row of A and jth column of B
"""
function flatfraginnerproduct!(A,B,C,n,p,i,j,nfrag,nlen)
    flatTCfrag!.(Ref(A),Ref(B),Ref(C),Ref(n),Ref(p),Ref(nfrag),Ref(i),1:nlen,1:nlen,Ref(j))
    return nothing
end

"""
flatmul_fragments(A,B,nfrag,p) performs A*B by partitioning it into tiles of size nfrag, where each entry involves p-double arithmetic
"""
function flatmul_fragments!(A,B,C,n,p,nfrag)
    @assert nfrag<=n && n%nfrag==0
    nlen = Int(n/nfrag)
    I,J = meshgrid(nlen)
    flatfraginnerproduct!.(Ref(A),Ref(B),Ref(C),Ref(n),Ref(p),I,J,Ref(nfrag),Ref(nlen))
    return nothing
end

"""
flatmatconv(A,B,C,n,p,f) uses matrix covolutions to compute A*B and fills it to C
"""
function flatmatconvhelp1!(A,B,C,n,p,i,j,k,f)
    Cp = zeros(n^2)
    f(A[i:p:end],B[j:p:end],Cp,n,1)
    C[k:p:end] += Cp
    return nothing
end

function flatmatconvhelp2!(A,B,C,n,p,k,f)
    flatmatconvhelp1!.(Ref(A),Ref(B),Ref(C),Ref(n),Ref(p),1:k,k:-1:1,Ref(k),Ref(f))
    return nothing
end

function flatmatconv!(A,B,C,n,p,f)
    flatmatconvhelp2!.(Ref(A),Ref(B),Ref(C),Ref(n),Ref(p),1:p,Ref(f))
    return nothing
end

"""
After performing the computations on the flat matrix A, flatnt(A,n,p) reforms it as an nxn matrix of p doubles 
"""
function flatnt(A,n,p)
    I,J = meshgrid(n)
    ent_A(i,j) = A[ent(n,p,i,j)]
    return ent_A.(I,J)
end
