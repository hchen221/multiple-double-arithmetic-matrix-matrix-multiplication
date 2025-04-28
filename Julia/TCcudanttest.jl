"""
TCcudatest.jl tests the matrix matrix multiplication using CUDA
"""

include("TCcudant.jl")

"""
test_pd_matrix() does a test run on the tiled matrix matrix multiplication with simulated TensorCore
"""
function test_pd_matrix(p=2,n=256,expmin=0,expmax=0)
    A = random_pd.(expmin*ones(n,n),expmax*ones(n,n),Ref(p))
    B = random_pd.(expmin*ones(n,n),expmax*ones(n,n),Ref(p))
    fA = flat(split4pd.(A))
    fB = flat(split4pd.(B))
    print("A,B in R^{$(n)x$(n)}, entries of $p-doubles\n\n")

    fC1 = zeros(Float64,n^2*4*p)
    fCc1 = zeros(Float64,n^2*4*p)
    t0 = time()
    flatTCKernel!(fA,fB,fC1,n,4*p)
    t_TCK = time()-t0
    function sampleTCK!(A,B,C,n,p,i,j)
        I,J = meshgrid(p)
        dotconv!.(Ref(A),Ref(B),Ref(C),Ref(n),Ref(p),Ref(i),Ref(j),I,J)
        return nothing
    end
    I,J = meshgrid(n)
    t0 = time()
    sampleTCK!.(Ref(fA),Ref(fB),Ref(fCc1),Ref(n),Ref(4*p),I,J)
    t_CU = time()-t0
    print("For direct inner product convolutions:\n    Vectorization took $t_TCK seconds\n    CUDA took $t_CU seconds\n    err=$(maximum(abs.(fCc1-fC1)))\n")

    fC2 = zeros(Float64,n^2*4*p)
    fCbuffer = zeros(Float64,n^2)
    fCc2 = zeros(Float64,n^2*4*p)
    t0 = time()
    flatmatconv!(fA,fB,fC2,n,4*p,flatTCKernel!)
    t_TCK2 = time()-t0
    t0 = time()
    matconv!(fA,fB,fCc2,n,4*p)
    t_CU2 = time()-t0
    print("For matrix convolutions:\n    Vectorization took $t_TCK2 seconds\n    CUDA took $t_CU2 seconds\n    err=$(maximum(abs.(Array(fCc2)-fC2)))\n")

    print("Cross errors:\n$(maximum(abs.(fC1-fC2)))\n$(maximum(abs.(fCc1-fCc2)))\n$(maximum(abs.(fCc1-fC2)))\n$(maximum(abs.(fCc2-fC1)))\n")
end

test_pd_matrix(2,64)
