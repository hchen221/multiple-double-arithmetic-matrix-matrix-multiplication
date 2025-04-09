"""
TCtest.jl tests the matrix matrix multiplication using makeshift TensorCore implementation
"""

include("TCflat.jl")

"""
test_pd_matrix() does a test run on the tiled matrix matrix multiplication with simulated TensorCore
"""
function test_pd_matrix(p=2,n=256,nfrag=16,expmin=0,expmax=0,direct=true)
    A = random_pd.(expmin*ones(n,n),expmax*ones(n,n),Ref(p))
    B = random_pd.(expmin*ones(n,n),expmax*ones(n,n),Ref(p))
    A8 = split4pd.(A)
    B8 = split4pd.(B)
    fA = flat(A8)
    fB = flat(B8)
    print("A,B in R^{$(n)x$(n)}, entries of $p-doubles\nTiles of size $nfrag\n\n")

    if direct
        C1 = zeros(4*n^2*p)
        C2 = zeros(4*n^2*p)
        t0 = time()
        flatTCKernel!(fA,fB,C1,n,4*p) # apply the TensorCore kernel to do each computation using double double arithmetic
        tC1 = time()-t0
        t0 = time()
        flatmatconv!(fA,fB,C2,n,4*p,flatTCKernel!) # the attempted matrix equivalent of double double multiplication
        tC2 = time()-t0
        print("C1 applies kernel directly (took $tC1 seconds)\nC2 uses split matrices (took $tC2 seconds)\nError of C1,C2 : $(maximum(abs.(C1-C2)))\n\n")
    end

    C1tiled = zeros(4*n^2*p)
    C2tiled = zeros(4*n^2*p)
    function mfrag!(A,B,C,n,p) # apply mul_fragments with given nfrag in outer scope for pass in purposes
        flatmul_fragments!(A,B,C,n,p,nfrag)
    end
    t0 = time()
    mfrag!(fA,fB,C1tiled,n,4*p)
    tC1tiled = time()-t0
    t0 = time()
    flatmatconv!(fA,fB,C2tiled,n,4*p,mfrag!)
    tC2tiled = time()-t0
    print("C1tiled applies tiling kernel directly (took $tC1tiled seconds)\nC2tiled uses split matrices (took $tC2tiled seconds)\nError of C1tiled,C2tiled : $(maximum(abs.(C1tiled-C2tiled)))\n\n")

    if direct
        print("Error between C1,C1tiled : $(maximum(abs.(C1-C1tiled)))\n")
        print("Error between C2,C2tiled : $(maximum(abs.(C2-C2tiled)))\n")
        print("\nAnd cross errors as a sanity check\n\n")
        print("Error between C1,C2tiled : $(maximum(abs.(C1-C2tiled)))\n")
        print("Error between C2,C1tiled : $(maximum(abs.(C2-C1tiled)))\n")
    end
end

test_pd_matrix(4,64,8)