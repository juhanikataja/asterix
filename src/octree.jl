module VDFOctreeApprox
#using Pkg; 
#Pkg.activate(".")
#Pkg.instantiate()
# using Debugger

using RegionTrees

using StaticArrays: SVector, SArray
using Profile
using PaddedViews

using TensorDecompositions

using Distributions
using Clustering

FloatType=Float32

module PolyBases

using Symbolics
using SpecialPolynomials
using StaticArrays: SVector

using CodecZlib

@variables x y z

FloatType=Float32
DEBUG=false
polyord = 2

function makebases(T,N)
    basispol = Legendre{T}
[basis(basispol,j)(x)*basis(basispol,k)(y)*basis(basispol,l)(z) for j in 0:N, k in 0:N, l in 0:N][:].|>Symbolics.toexpr
end

function cosbases(N)
[cos(j*pi*x)*cos(k*pi*y)*cos(l*pi*z) for j in 0:N, k in 0:N, l in 0:N][:].|>Symbolics.toexpr
end


#bases = makebases(Rational{Int64},polyord)
basesf = makebases(FloatType,polyord)
# basesf=cosbases(polyord)

basislen = length(basesf)

eval(Expr(:function,:(phi(x,y,z)), quote SVector($(basesf...),) end))

bodystring = foldl(*,("phiwrk[$(i)] = "*string(Meta.parse(string(basesf[i])))*"\n" for i in axes(basesf,1)))

eval(Meta.parse(
    "function phi!(x::FloatType,y::FloatType,z::FloatType,phiwrk::Vector{FloatType})\n"*
        bodystring*
        "nothing\n"*
        "end\n"))

if DEBUG
    println(bodystring)
end
bdim = size(basesf,1)

end



using .PolyBases

phifun3=PolyBases.phi

eval(Meta.parse(
    "function phifun3!(x::T,y::T,z::T,phiwrk::Vector{T}) where {T}\n"*
        foldl(*,("phiwrk[$(i)] = "*string(Meta.parse(string(PolyBases.basesf[i])))*"\n" for i in axes(PolyBases.basesf,1)))*
        "nothing\n"*
        "end\n"))

function getdims_gen(solu::Cell{T,L,S,M}, kuva, root::Cell{T,L,S,M}) where {T,L,S,M}
    r0 = 1 .+ (solu.boundary.origin./root.boundary.widths) .* size(kuva) .|> ceil.|> Int64
    r1 = ((solu.boundary.origin+solu.boundary.widths).*size(kuva)./(root.boundary.widths)) .|> ceil .|> Int64 
    spans = [r0[k]:r1[k] for k in axes(r0,1)]
    h = solu.boundary.widths ./ length.(spans)
    spans, h
end

# TODO: these assume unit cuboid solu
function getdims(solu::Cell{T,2,S,M}, kuva) where {T,S,M}
    r0 = 1 .+ solu.boundary.origin .* size(kuva) .|> ceil.|> Int64
    r1 = ((solu.boundary.origin+solu.boundary.widths).*size(kuva)) .|> ceil .|> Int64
    spans = SVector(r0[1]:r1[1], r0[2]:r1[2])
    #[r0[k]:r1[k] for k in 1:length(r0)]
    h = solu.boundary.widths ./ length.(spans)
    spans, h
end


# TODO: these assume unit cuboid solu
function getdims(solu::Cell{T,3,S,M}, kuva) where {T,S,M}
    r0 = 1 .+ solu.boundary.origin .* size(kuva) .|> ceil.|> Int64
    r1 = ((solu.boundary.origin+solu.boundary.widths).*size(kuva)) .|> ceil .|> Int64
    #spans = [r0[k]:r1[k] for k in 1:length(r0)]
    spans = SVector(r0[1]:r1[1], r0[2]:r1[2], r0[3]:r1[3])
    h = solu.boundary.widths ./ length.(spans)
    spans, h
end



function approx(solu::Cell{T,2,S,M}, kuva) where {T,S,M}
    center = 0.5
    dims = getdims(solu,kuva)
    h = prod(dims[2])
    phi0 = phifun(0.0,0.0)
    A = zeros(size(phi0,1), size(phi0,1))
    b = zeros(size(phi0))
    bias = first.(dims[1]) .-1
    scale = 1. ./ length.(dims[1]) 
    for m in dims[1][1]
        x = scale[1]*(m-center-bias[1])#(m - bias[1] - 0.5) * dims[2][1]
        #display(x)
        for n in dims[1][2]
            y = scale[2]*(n-center-bias[2])#(n - bias[2] - 0.5) * dims[2][2]
            phi = phifun(x,y)
            b = b + phi * FloatType(kuva[m,n])*h
            A = A + phi * phi' * h # TODO calculate this exactly
        end
    end
    #A = h*fmat/scale[1]/scale[2]/4
    A,b
end

function approx!(solu::Cell{T,3,S,M}, kuva, A, b, phiwrk) where {T,S,M}
    e = one(FloatType)
    center = FloatType(0.5)
    spans, _ = getdims(solu,kuva)
    bias = first.(spans) .-1
    scale = e ./ length.(spans) 
    A .= zero(A[1])
    b .= zero(b[1])
    for k in spans[3]
        z = 2*scale[3]*(k-center-bias[3]) - e

        for n in spans[2]
            y = 2*scale[2]*(n-center-bias[2])  - e

            for m in spans[1] 
                x = 2*scale[1]*(m-center-bias[1]) - e
                phifun3!(x,y,z,phiwrk)
                for F in axes(b,1)
                    b[F] = b[F] + phiwrk[F]*kuva[m,n,k]
                    for Q in axes(b,1)
                        A[Q,F] = A[Q,F] + phiwrk[Q]*phiwrk[F]
                    end
                end
            end
        end
    end
    nothing
end

function approx(solu::Cell{T,3,S,M}, kuva)  where {T,S,M}
    center = 0.5
    dims = getdims(solu,kuva)
    h = prod(dims[2])
    phi0 = phifun3(0.0,0.0,0.0)
    phidim = size(phi0,1)
    A = zeros(phidim, phidim)
    b = zeros(phidim)
    bias = first.(dims[1]) .-1
    scale = 1. ./ length.(dims[1]) 
    for k in dims[1][3]
        z = 2*scale[3]*(k-center-bias[3]) -1.

        for n in dims[1][2]
            y = 2*scale[2]*(n-center-bias[2])-1.

            for m in dims[1][1] 
                x = 2*scale[1]*(m-center-bias[1])-1. #(m - bias[1] - 0.5) * dims[2][1]
                phi = phifun3(x,y,z)
                b = b + phi * FloatType(kuva[m,n,k])
                A = A + phi * phi'  
            end

        end
    end
    A*h,b*h
end

function accumulate_cell!(solu::Cell{T,3,K,M}, phiwrk, kuva) where {T,K,M}
  c = solu.data.c
  e = one(FloatType)
  center = FloatType(0.5)
  span, _ = getdims(solu,kuva)
  bias = first.(span) .-1
  scale = e ./ length.(span) 

  for k in span[3]
    rk = k-bias[3]
    z = 2*scale[3]*(k-center-bias[3]) - e
    for n in span[2]
      rn = n-bias[2]
      y = 2*scale[2]* (n - center -bias[2]) - e
      for m in span[1]
        rm = m-bias[1]
        x = 2*scale[1] * (m - center - bias[1]) - e
        phifun3!(x,y,z,phiwrk)
        kuva[m,n,k] = kuva[m,n,k] + c'*phiwrk
      end
    end
  end

end

function improve_residual!(solu::Cell{T,3,K,M}, A, b, phiwrk, residual) where {T,K,M}
    e = one(FloatType)
    center = FloatType(0.5)

    c = A\b
    span, hs = getdims(solu,residual)
    h = prod(hs)

    bias = first.(span) .- 1
    scale = e ./ length.(span) 
    for k in span[3]
        rk = k-bias[3]
        z = 2*scale[3]*(k-center-bias[3]) - e

        for n in span[2]
            rn = n-bias[2]
            y = 2*scale[2]* (n - center -bias[2]) - e

            for m in span[1]
                rm = m-bias[1]
                x = 2*scale[1] * (m - center - bias[1]) - e
                phifun3!(x,y,z, phiwrk)

                residual[m,n,k] = residual[m,n,k] - c'*phiwrk
            end
        end
    end
    nothing

end

function calcerror!(solu::Cell{T,3,K,M}, A, b, phiwrk, kuva, reconstr; alpha=1.0, beta=1.0, nu=4) where {T,K,M}
    e = one(FloatType)
    center = FloatType(0.5)
    # c = lsqr(A,b; maxiter=length(b))
    c = A\b
    span, hs = getdims(solu,kuva)
    h = prod(hs)

    bias = first.(span) .- 1
    scale = e ./ length.(span) 
    err = zero(FloatType)
    hsq = sqrt(h)

    kuva_scale = zero(FloatType)
    localmax = zero(FloatType)

    # Threads.@threads 
    for k in span[3]
        rk = k-bias[3]
        z = 2*scale[3]*(k-center-bias[3]) - e #(n - bias[2] - 0.5) * dims[2][2]

        for n in span[2]
            rn = n-bias[2]
            y = 2*scale[2]* (n - center -bias[2]) - e

            for m in span[1]
                rm = m-bias[1]
                x = 2*scale[1] * (m - center - bias[1]) - e

                phifun3!(x,y,z, phiwrk)
                reconstr[rm,rn,rk] = c'*phiwrk

                cterm = kuva[m,n,k]-reconstr[rm, rn, rk]

                err = err + alpha*h*(cterm^nu)

                localmax = maximum((localmax, abs(cterm)))

                # Do backward diff since the data should exist, skip boundary planes rk, rn, or rm = 0
                if rk>1 err = err + beta * hsq * ((cterm-kuva[m,n,k-1]+reconstr[rm,rn,rk-1])^nu) end
                if rn>1 err = err + beta * hsq * ((cterm-kuva[m,n-1,k]+reconstr[rm,rn-1,rk])^nu) end
                if rm>1 err = err + beta * hsq * ((cterm-kuva[m-1,n,k]+reconstr[rm-1,rn,rk])^nu) end

                # kuva_scale = kuva_scale + h*(abs(kuva[m,n,k])^nu)
                kuva_scale = maximum((abs(kuva_scale), abs(kuva[m,n,k])))

            end
        end
    end

    err = err # / (kuva_scale^(1.0/nu))

    FloatType(err), FloatType(kuva_scale), FloatType(localmax)
end

function calcerror(solu::Cell{T,3,K,M}, A, b, kuva;alpha=1.0, beta=1000.0, nu=4) where {T,K,M}
    center = 0.5
    # c = zeros(size(b))
    # Awrk = zeros(size(A))
    # bwrk = zeros(size(b))
    # Awrk .= A
    # bwrk .= b
    # lsqr!(c, Awrk, bwrk; maxiter=size(A,2))
    c = A\b
    dims = getdims(solu,kuva)
    h = prod(dims[2])

    reconstr = zeros((length.(dims[1]))...)
    bias = first.(dims[1]) .-1
    scale = 1. ./ length.(dims[1]) 
    #err = zero(FloatType)
    #energy = zero(FloatType)

    # Threads.@threads 
    for k in dims[1][3]
        z = 2*scale[3]*(k-center-bias[3])-1#(n - bias[2] - 0.5) * dims[2][2]
        for n in dims[1][2]
            y = 2*scale[2]* (n - center -bias[2]) - 1
            for m in dims[1][1]
                x = 2*scale[1] * (m - center - bias[1]) -1
                val = phifun3(x,y,z)
                reconstr[m-bias[1],n-bias[2],k-bias[3]] = c'*val
            end
            #err = err + h*(kuva[m,n]-reconstr[m-bias[1],n-bias[2]])^2
            #energy = h*(0.1 + kuva[m,n])^2
        end
    end

    S = kuva[dims[1]...] .- reconstr
    #R = (kuva[dims[1]...] .+ reconstr .+ 1e-6)./2
    err = (alpha*h * sum(S.^nu) .+
        beta*sqrt(h)*sum(diff(S,dims=1).^nu) + sum(diff(S,dims=2).^nu) + sum(diff(S, dims=3).^nu))^(1.0/nu)

    #energy = h*sum((0.1 .+ kuva[dims[1]...]).^2)

    reconstr, FloatType(err) #/ sqrt(FloatType(energy))
end

function calcerror(solu::Cell{T,2,L,M}, A, b, kuva;alpha=1.0, beta=1000.0, nu=4) where {T,L,M}
    center = 0.5
    c = A \ b

    dims = getdims(solu,kuva)
    h = prod(dims[2])

    reconstr = zeros((length.(dims[1]))...)
    bias = first.(dims[1]) .-1
    scale = 1. ./ length.(dims[1]) 
    #err = zero(FloatType)
    #energy = zero(FloatType)

    for m in dims[1][1]
        x = scale[1] * (m - center - bias[1]) 
        for n in dims[1][2]
            y = scale[2]* (n - center -bias[2]) 
            val = phifun(x,y)
            reconstr[m-bias[1],n-bias[2]] = c'*val
            #err = err + h*(kuva[m,n]-reconstr[m-bias[1],n-bias[2]])^2
            #energy = h*(0.1 + kuva[m,n])^2
        end
    end

    S = kuva[dims[1]...] .- reconstr
    err = (alpha*h * sum((kuva[dims[1]...] .- reconstr).^nu) +
        beta*sqrt(h)*sum(diff(S,dims=1).^nu) + sum(diff(S,dims=2).^nu))^(1.0/nu)

    #energy = h*sum((0.1 .+ kuva[dims[1]...]).^2)

    reconstr, FloatType(err) #/ sqrt(FloatType(energy))
end

function approx_and_recon(solu,kuva; alpha=1.0, beta=1.0, nu=4)
    A, b = approx(solu,kuva)
    reconstr, err = calcerror(solu, A, b, kuva; alpha=alpha, beta=beta, nu=nu)
end

function approx_and_recon!(solu,kuva, A, b, phiwrk;reconstr=nothing, alpha=1.0, beta=1.0, nu=4)
    approx!(solu,kuva, A, b, phiwrk)

    reco = if isnothing(reconstr) 
        span, _ = getdims(solu,kuva)
        similar(kuva[span...])
    else
        span, _ = getdims(solu,kuva)
        @view reconstr[span...]
    end
    calcerror!(solu, A, b, phiwrk, kuva, reco; alpha=alpha, beta=beta, nu=nu)
end

function improveold!(solu::Cell{T,N,S,M}, uudelleen, kuva) where {T,N,S,M}
    rec_new, err = approx_and_recon(solu, kuva)
    rec_new = rec_new
    dims = getdims(solu,kuva)
    uudelleen[dims[1]...] = rec_new
    err
end

function improve!(solu::Cell{T,N,S,M}, reconstr, kuva, A, b, phiwrk) where {T,N,S,M}
    #span,_ = getdims(solu, kuva)
    #reco = @view reconstr[span...]

    err = approx_and_recon!(solu, kuva, A, b, phiwrk; reconstr=reconstr)
    err
end

"""
W is the original padded size of the image (padded to 3*2^k)
"""
function getdims_new(solu::Cell{T,3,S,M}, W) where {T,S,M}
    r0 = 1 .+ solu.boundary.origin .* W .|> ceil .|> Int64
    r1 = ((solu.boundary.origin+solu.boundary.widths) .* W) .|> ceil .|> Int64
    SVector(r0[1]:r1[1], r0[2]:r1[2], r0[3]:r1[3])
end

function approx_new!(solu::Cell{T,3,S,M}, kuva, A, b, phiwrk) where {T,S,M}
  center = 0.5
  spans, _ = getdims(solu,kuva)
  bias = first.(spans) .-1
  scale = 1. ./ length.(spans) 
  for k in spans[3]
    z = 2*scale[3]*(k-center-bias[3]) -1.

    for n in spans[2]
      y = 2*scale[2]*(n-center-bias[2]) -1.

      for m in spans[1] 
        x = 2*scale[1]*(m-center-bias[1]) -1.
        phifun3!(x,y,z,phiwrk)
        for F in axes(b,1)
          b[F] = b[F] + phiwrk[F]*kuva[m,n,k]
          for Q in axes(b,1)
            A[Q,F] = A[Q,F] + phiwrk[Q]*phiwrk[F]
          end
        end
      end
    end
  end
  nothing
end


mutable struct Leaf_t
    err::FloatType
    c::Vector{FloatType}
    act::Bool
end

struct TLeaf_t
  D::Union{Nothing, Tucker}
  active::Bool
  residual
end

function improve_residual_tucker!(solu::Cell{T,3,K,M}, residual;tucker_core_dim=(1,1,1)) where {T,K,M}

  spans, _ = getdims(solu, residual)
  f_spans = [sp[1] - 1 for sp in spans]

  solu.data = 
    let D = hosvd(Array(residual[spans...]), tucker_core_dim), 
      ftype = typeof(residual[1]),
      res_sum = zero(ftype)

      for i1 in spans[1]
        I1 = i1 - f_spans[1]
        for i2 in spans[2]
          I2 = i2 - f_spans[2]
          for i3 in spans[3]
          I3 = i3 - f_spans[3]
            acc = zero(ftype)
            for j1 in Base.OneTo(tucker_core_dim[1])
              for j2 in Base.OneTo(tucker_core_dim[2])
                for j3 in Base.OneTo(tucker_core_dim[3])
                  acc = acc + D.core[j1,j2,j3] * D.factors[1][I1,j1]*D.factors[2][I2,j2]*D.factors[3][I3,j3]
                end
              end
            end
            residual[i1,i2,i3] = residual[i1,i2,i3] - acc
            res_sum = maximum([res_sum, residual[i1,i2,i3]|>abs])
            # res_sum = res_sum + residual[i1,i2,i3]|>abs
          end
        end
      end
      #res_sum = sqrt(res_sum) / prod(length.(spans))
      # res_sum = res_sum / prod(length.(spans))
      TLeaf_t(D, solu.data.active, res_sum)

    end

end

function reconstruct!(reco::Array{T,3}, D, spans; tucker_core_dim=(1,1,1)) where T
  f_spans = [sp[1] - 1 for sp in spans]
  for i1 in spans[1]
    I1 = i1 - f_spans[1]
    for i2 in spans[2]
      I2 = i2 - f_spans[2]
      for i3 in spans[3]
        I3 = i3 - f_spans[3]
        acc = zero(T)
        for j1 in Base.OneTo(tucker_core_dim[1])
          for j2 in Base.OneTo(tucker_core_dim[2])
            for j3 in Base.OneTo(tucker_core_dim[3])
              acc = acc + D.core[j1,j2,j3] * D.factors[1][I1,j1]*D.factors[2][I2,j2]*D.factors[3][I3,j3]
            end
          end
        end
        reco[i1,i2,i3] = reco[i1,i2,i3] + acc
      end
    end
  end
end

function reconstruct!(tree::Cell{T, 3, K, M}, reco) where {T,K,M}
    if tree.data.active 
        spans, _ = getdims(tree, reco)
        reconstruct!(reco, tree.data.D, spans; tucker_core_dim=size(tree.data.D.core))
    end
end

function pad_and_scale(img; bdim=PolyBases.bdim, simple=false)
  L = round(bdim^(1/3))|>UInt64

  spans = size(img) # return this

  spanranges = [Base.OneTo(spans[i]) for i in 1:3]
  k = ceil(log2(maximum(spans) ./ L))
  W = if simple maximum(spans) else L*2^k end

  shift = minimum(img[:]) # return this
  for k in axes(img[:],1)
    img[k] = img[k] - shift
  end

  scale = maximum(img[:]) # return this
  for k in axes(img[:],1)
    img[k] = img[k] / scale
  end

  cell = Cell(SVector(0., 0., 0.), SVector(1., 1., 1.), nothing)

  pad_span = getdims_new(cell, W)
  padimg = PaddedView(zero(img[1]), img, (pad_span[1], pad_span[2], pad_span[3])) # return this

  Array(padimg), shift, scale, spanranges

end

function compress_tucker(img; maxiter=100, tol=1e-3, tucker_core_dim=(1,1,1), verbose=true, nbits=10)

  residual, shift, scale, spanranges = pad_and_scale(img|>copy ; bdim=prod(tucker_core_dim), simple=false)
  padimg = copy(residual)

  tree = Cell(SVector(0., 0., 0.), SVector(1., 1., 1.), TLeaf_t(nothing, true, residual[:].^2 |> sum |> sqrt))
  

  for iter in Base.OneTo(maxiter)

    maxres = zero(FloatType)
    for leaf in allleaves(tree)
      maxres = maximum((leaf.data.residual, maxres))
    end

    if maxres/tree.data.residual < tol
      break
    end

    if verbose println("$(iter)\t: maxres = $(maxres / tree.data.residual)") end

    for leaf in allleaves(tree)
      leaf.data = TLeaf_t(leaf.data.D, false, leaf.data.residual)
      if !(leaf.data.residual == maxres) continue end

      improve_residual_tucker!(leaf, residual;tucker_core_dim=tucker_core_dim)
      split!(leaf)
      leaf.data = TLeaf_t(leaf.data.D, true, leaf.data.residual)
    end

  end

  sizes = mapreduce(x-> if (x.data.active) Array([1,sum(prod.(size.(x.data.D.core)))+sum(prod.(size.(x.data.D.factors)))]) else Array([0,0]) end, +, allcells(tree))

  active_cells = filter(x->x.data.active, collect(allcells(tree)))
  packed = pack_cells(active_cells; nbits=nbits)
  #size_bytes = sum(sizeof.([packed...]))
  size_bytes = get_packed_size(packed)

  unpacked = unpack_cells(packed...)
  residual .= 0.0
  map(unpacked) do cell
    reconstruct!(cell, residual)
  end
  residual = padimg - residual

  residual[spanranges...] .* scale, scale.*padimg[spanranges...] .+ shift, active_cells, size_bytes

end

function compress_conservative(img; maxiter=100, tol=1e-3, verbose=true, errtype="ltwo")

    bdim = size(PolyBases.basesf,1)
    A = zeros(FloatType, bdim, bdim)
    b = zeros(FloatType, bdim)
    phiwrk = similar(b)

    L = round(bdim^(1/3))|>Int64
    spans = size(img)
    spanranges = [1:spans[i] for i in 1:3]
    k = ceil(log2(maximum(spans) ./ L))
    W = (PolyBases.polyord+1)*2^k 

    cell = Cell( SVector(0., 0., 0.), SVector(1., 1., 1.), Leaf_t(0.0, zeros(size(b)), false) )

    shift = minimum(img[:])
    println("shift: $(shift)")
    img = Array{FloatType}(img .- shift)

    scale = maximum(img[:])
    println("scale: $(scale)")
    img = Array{FloatType}(img ./ scale)

    pad_span = getdims_new(cell, W)
    padimg = PaddedView(zero(img[1]), img, (pad_span[1], pad_span[2], pad_span[3]))

    span, _ = getdims(cell,padimg)
    residual=zeros(span...)
    residual[:,:,:] = padimg[:,:,:]

    for iter in 1:maxiter 
        # maxres, Ires = findmax(abs.(residual))# ./ floorpadimg))
        maxres = zero(FloatType)

        for leaf in allleaves(cell)
            ldims, h = getdims(leaf, padimg)
            if errtype == "ltwo"
                err =  (prod(h) *sum(abs.(residual[ldims...]).^8)).^(1/8)
            else
                err = maximum(residual[ldims...][:] .|> abs)
            end
            leaf.data = Leaf_t(err, leaf.data.c, leaf.data.act)
            # println(leaf.data.err)
            maxres = maximum((leaf.data.err, maxres))
            # println("at first: $(leaf.data.err)")
        end

        if maxres < tol 
            break
        end

        if verbose println("$(iter)\t: maxres = $(maxres)") end

        for leaf in allleaves(cell)

            # ldims = getdims_new(leaf, W)
            leaf.data.act = false
            # println("after: $(leaf.data.err)")

            # if !((Ires[1] in ldims[1]) & (Ires[2] in ldims[2]) & (Ires[3] in ldims[3])) continue end
            if !(leaf.data.err == maxres) continue end

            approx!(leaf, residual, A, b, phiwrk)
            improve_residual!(leaf, A, b, phiwrk, residual)
            split!(leaf)
            leaf.data = Leaf_t(leaf.data.err, A\b, true)
        end
    end

    let active_cells = filter(x->x.data.act, collect(allcells(cell)))
        orig_dense_size = prod(spanranges .|> x->length(x))
        compression_ratio =  4*orig_dense_size/(4*length(active_cells)*length(cell.data.c) + 3*length(active_cells))
        residual[spanranges...] .* scale, scale.*padimg[spanranges...] .+ shift, active_cells, compression_ratio
    end

end

function compress(img; maxiter=100, nu=4, tol=1e-4, alpha=1.0, beta=1.0, verbose=true)
    bdim = size(PolyBases.basesf,1)
    A = zeros(FloatType, bdim, bdim)
    b = zeros(FloatType, bdim)
    phiwrk = similar(b)

    L = round(bdim^(1/3))|>Int64
    spans = size(img)
    k = ceil(log2(maximum(spans) ./ L))
    W = (PolyBases.polyord+1)*2^k 


    cell = Cell( SVector(0., 0., 0.), SVector(1., 1., 1.), 0.0 )
    scale = maximum(img[:])
    img = Array{FloatType}(img ./ scale)
    pad_span = getdims_new(cell, W)
    padimg = PaddedView(zero(img[1]), img, (pad_span[1], pad_span[2], pad_span[3]))

    span, _ = getdims(cell,padimg)
    reconstr=zeros(span...)

    initial_error = 0.0

    for iter in 1:maxiter
        tot_err = zero(FloatType)
        maxerr = zero(FloatType)
        for leaf in allleaves(cell)
            approx!(leaf, padimg, A, b, phiwrk)
            span, _ = getdims(leaf, padimg)

            err, kuva_scale, localmax = calcerror!(leaf, A, b, phiwrk, padimg, @view reconstr[span...]; alpha=alpha, beta=beta, nu=nu)
            # leaf.data = err
            localmax = localmax/(kuva_scale+FloatType(tol))
            leaf.data = localmax
            tot_err = tot_err + err


# maxerr = if err > maxerr err else maxerr end
maxerr = maximum((localmax, maxerr)) # if localmax > maxerr localmax else maxerr end

            if span[1] == L leaf.data = -1.0 end

        end
if iter == 1 initial_error = maxerr end

if verbose
            print((maxerr/initial_error))
            # print((maxerr/initial_error).^(1. /nu))
            print("\t")
            println(maxerr)
end

        for leaf in allleaves(cell)
            if leaf.data == maxerr 
                split!(leaf) 
            end
        end
        # if (tot_err/initial_error).^(1. /nu) < tol
        if maxerr< tol #/initial_error < tol
            break
        end
    end

    A, b, scale .* img, scale .* reconstr[(1:spans[j] for j in 1:3)...], cell, 
    map(collect(allleaves(cell))) do leaf
        approx!(leaf, padimg, A,b, phiwrk)
        spans, _ = getdims(leaf, padimg)
    (leaf, A\b, spans)
    end

end

function testapprox(spans::Tuple{Int64,Int64,Int64}; fillfun=nothing, maxiter=5, nu=4, tol=1e-3, alpha=1.0, beta=1.0)

    bdim = size(PolyBases.basesf,1)
    A = zeros(bdim, bdim)
    b = zeros(bdim)
    phiwrk = similar(b)

    L = round(bdim^(1/3))|>Int64
    k = ceil(log2(maximum(spans) ./ L))
    W = 3*2^k 


    cell = Cell( SVector(0., 0., 0.), SVector(1., 1., 1.), 0.0 )

    pad_span = getdims_new(cell, W)

    img = zeros(spans)
    if !isnothing(fillfun)
        for J in eachindex(view(img,(1:spans[j] for j in 1:3)...))
            img[J] = fillfun(J[1], J[2], J[3])
        end
    end
    padimg = PaddedView(zero(img[1]), img, (pad_span[1], pad_span[2], pad_span[3]))

    span, _ = getdims(cell,padimg)
    reconstr=zeros(span...)

    initial_error = 0.0
    for iter in 1:maxiter
        tot_err = zero(FloatType)
        maxerr = zero(FloatType)
        for leaf in allleaves(cell)
            approx!(leaf, padimg, A, b, phiwrk)
            span, _ = getdims(leaf, padimg)

            err = calcerror!(leaf, A, b, phiwrk, padimg, @view reconstr[span...]; alpha=alpha, beta=beta, nu=nu)
            leaf.data = err
            tot_err = tot_err + err
            maxerr = if err > maxerr err else maxerr end

            if span[1] == L leaf.data = -1.0 end

        end
        if iter == 1 initial_error = tot_err end


        print((tot_err/initial_error).^(1. /nu))
        print("\t")
        println(maxerr.^(1. /nu))

        for leaf in allleaves(cell)
            if leaf.data == maxerr 
                split!(leaf) 
            end
        end
        if (tot_err/initial_error).^(1. /nu) < tol
            break
        end
    end


    A, b, img, reconstr[(1:spans[j] for j in 1:3)...], cell, 
    map(collect(allleaves(cell))) do leaf
        approx!(leaf, padimg, A,b, phiwrk)
        spans, _ = getdim(leaf, padimg)
    (leaf, A\b, spans)
    end

#reco = reconstr[(1:spans[j] for j in 1:3)...]
end

function visualcompare(reconstr, img; slice=5, epsterm=1e-3)

    fig = Makie.Figure()

    ax, imleft = heatmap(fig[1,1], reconstr[:,:,slice], interpolate=false)
    ax, imright = heatmap(fig[1,2], img[:,:,slice], interpolate=false)
    ax, imrightright = heatmap(fig[1,3], (img[:,:,slice]-reconstr[:,:,slice])./(img[:,:,slice].+reconstr[:,:,slice].+epsterm), interpolate=false)
    Colorbar(fig[1,1][1,2], imleft)
    Colorbar(fig[1,2][1,2], imright)
    Colorbar(fig[1,3][1,2], imrightright)

    fig

end

function hosvd_compress(kuva, core_size)
  core_size = (core_size[1], core_size[2], core_size[3])
  println(core_size)
  println(typeof(core_size))
  println(typeof(kuva))
  kuva = Array{FloatType}(kuva)

  D = hosvd(kuva, core_size)
  reco = similar(kuva)
  fill!(reco, zero(typeof(kuva[1])))
  for i1 in axes(kuva,1)
    for i2 in axes(kuva,2)
      for i3 in axes(kuva,3)
        for j3 in Base.OneTo(core_size[3])
          for j2 in Base.OneTo(core_size[2])
            for j1 in Base.OneTo(core_size[1])
              reco[i1,i2,i3] = reco[i1,i2,i3] + D.core[j1,j2,j3] * D.factors[1][i1,j1]*D.factors[2][i2,j2]*D.factors[3][i3,j3]
            end
          end
        end
      end
    end
  end
  kuva_size = size(kuva)|>prod
  reco_size = prod(size(D.core)) + prod(size(D.factors[1]))+ 
    prod(size(D.factors[2]))+ prod(size(D.factors[3]))

  reco, kuva_size/reco_size
end



function quantize_array(IntType, X::Array{T}) where T
  limits = (minimum(X), maximum(X))
  scale = (limits[2]-limits[1])
  center = (limits[1]+limits[2])/2
  half = if typemin(IntType) < 0 scale = scale * 2; T(0.0) else T(0.5) end
  Y= let Xbig = Float64.(X)
    IntType.( round.( (typemax(IntType) -1) .* (((Xbig.-center) ./ scale) .+ half) ) )
  end
  Y, (scale, center)
end

function unquantize_array(FloatType, Y::Array{T}, S) where {T}
  scale = FloatType(S[1])
  center = FloatType(S[2])
  half = if typemin(T) < 0 FloatType(0.0) else FloatType(0.5) end
  X =  (((FloatType.(Y) ./ typemax(T)) .- half) .* scale) .+ center
  X
end

function serialize_leaves(leaves::Vector{Cell{A,B,C,D}}) where {A,B,C,D}
  # Assume equal sized core dimensions
  core_dims = size(leaves[1].data.D.core).|>UInt8
  core_size = prod(core_dims)


  factor_lengths = map(leaves) do leaf
     UInt16(sum(prod.(size.(leaf.data.D.factors))))
  end
  factor_inds = cumsum(vcat([1], factor_lengths))

  DType = typeof(leaves[1].data.D.core[1])
  ser_len = length(leaves)*core_size + sum(factor_lengths)
  serialized = Vector{DType}(undef, ser_len)

  for L in axes(leaves,1)
    serialized[core_size*(L-1) .+ (1:core_size)] = leaves[L].data.D.core[1:core_size]
  end
  bias = core_size * length(leaves)

  allfactors = [vcat((leaf.data.D.factors .|> x->x[:])...) for leaf in leaves]

  for L in 1:(length(factor_inds)-1)
    serialized[bias .+ (factor_inds[L]:factor_inds[L+1]-1)] = allfactors[L]
  end
  serialized, core_dims, factor_lengths, length(leaves)
end

function deserialize_leaves(serialized, core_dims, factor_lengths, n_leaves)
  L_core = 1:(n_leaves*prod(core_dims))
  cores = reshape(serialized[L_core], Tuple(vcat([core_dims...], [n_leaves])))
  cores
  
  factor_sizes = let C = sum(core_dims)
    map(factor_lengths) do L
      l = UInt16(L/C)
    [ [l,cd] for cd in core_dims ]
    end
  end
  factors = map(factor_sizes) do fs
    zeros(fs[1]...,3)
  end
  bias = n_leaves*prod(core_dims)
  for F in axes(factors,1)
    factors[F][:,:,:] = reshape(serialized[(bias+1):(factor_lengths[F]+bias)], size(factors[F])...)
    bias = bias+factor_lengths[F]
  end
  factor_sizes
  cores,factors

end

function f2bin(x; maxdepth=8)
    B = BitArray(undef, 8)*false
    y = x
    for iter in 1:maxdepth
        B[iter] = if y-0.5 >= 0 true else false end
        y = (y-B[iter]/2)*2
        if y == 0
            break
        end
    end
    B
end

function cell_to_binary(cell)
    bnd = [cell.boundary.origin, cell.boundary.widths]
    centroids = bnd[1].+(bnd[2])./2
    f2bin.(centroids)
end

function bin2f(B)
    depth = 0
    y = Float64(0.0)
    for n in axes(B,1)
        if sum(B[n:end]) == 0 
            depth = n-1
            break
        end
        y = y+(B[n] * (2.0 ^(-n)))
    end
    left = y-2.0^(-depth)
    right = y+2.0^(-depth)
    return [left,right-left]
end

function tobitvector(Q;nbits=8) 
    BitVector(digits(UInt(Q), base=2, pad=nbits))[1:nbits]
end

function bv_touint(Q::BitVector, UIType)
    acc = zero(UIType)
    for n in axes(Q,1)
        acc = acc + UIType(Q[n]) * UIType(2^(n-1))
    end
    acc
end

function asbytes(Q::BitVector)
    [bv_touint(Q[(bn*8+1):((bn+1)*8)], UInt8) for bn in 0:(div(length(Q),8)-1)]
end

function ReLU(z)
    if z > zero(0)
        z
    else
        zero(z)
    end
end

function quantize_clustering(X::Vector{T}, nbits::UInt, K::UInt) where T

    N = 2^nbits
    kmeans_init = if K==1
        [findmin(abs.(quantile(X,0.5) .- X))[2]]
    else
         [findmin(abs.(qx .- X))[2] for qx in quantile(X, range(0, 1; length=K))]
    end
    xclust = kmeans(X', K; init=kmeans_init, tol=1e-8, maxiter=2000)#, display=:iter)

    sorted = sortperm(xclust.centers[:])

    VM = map(sorted) do i
        C = [minimum(X[xclust.assignments.==i]), maximum(X[xclust.assignments.==i])]
        center = xclust.centers[i]
        bdist = maximum(abs.(C .- center))
        (center - bdist, center + bdist, bdist * 2, length(X[xclust.assignments.==i]))
    end

    beta = N / sum((y[3] * y[4] for y in VM))
    alpha = (y[3] * y[4] for y in VM)
    N0 = beta .* collect(alpha)
    N_clust = UInt64.(round.(N0))
    N_clust[N_clust.==0] .= one(UInt64)

    while N > sum(N_clust)
        errs = alpha ./ N_clust
        _, ind = findmax(errs)
        N_clust[ind] = N_clust[ind] + 1
    end
    while sum(N_clust) > N
        errs = alpha ./ N_clust
        _, ind = findmin((
            if N_clust[i] == 1
                Inf
            else
                errs[i]
            end
            for i in axes(errs, 1)))
        N_clust[ind] = N_clust[ind] - 1
    end


    nq = map(X) do z
        _, I = findmin((ReLU(VM[ind][1] - z) + ReLU(z - VM[ind][2]) for ind in axes(VM, 1)))
        bias = sum(N_clust[1:(I-1)])
        if N_clust[I] == 1
            bias + 1
        else
            a = VM[I][1]
            ba = VM[I][3]
            N = N_clust[I]
            n = bias + maximum((1, minimum((N, Int(ceil(((z - a) / ba + 1 / (2 * N)) * N))))))
            n
        end
    end

    R_limits = map(VM) do L
        L[1],L[2]
    end
  numbits = length(nq)*nbits
  pad = BitVector(zeros(8-numbits+((div(numbits,8))*8)))
  vcat(tobitvector.(nq .- 1;nbits=nbits)..., pad)|>asbytes, R_limits, N_clust, nbits, length(pad)
end

function dequantize_clustering(bytes, R_limits, N_clust, nbits, n_pad)
    Y = vcat(tobitvector.(bytes;nbits=8)...)
    num_numbers = div(length(Y[1:(end-n_pad)]), nbits)

    cs_clust = cumsum(vcat([zero(typeof(N_clust[1]))], N_clust))[1:(end-1)]
    map(0:(num_numbers-1)) do bn
        n = bv_touint(Y[(bn*nbits+1):(bn*nbits+nbits)],UInt64) + 1
        I = sum(n .> cs_clust)
        bias = cs_clust[I]
        if N_clust[I] == 1
            R_limits[I][1]
        else
            a = R_limits[I][1]
            ba = R_limits[I][2]-R_limits[I][1]
            N = N_clust[I]
            nn = n-bias
            (nn / N - 1 / (2 * N)) * ba + a
        end
    end
end

using CodecZlib
function pack_cells(cells::Array; nbits=8)
    serialized, core_dims, factor_lengths, n_leaves = serialize_leaves([cell for cell in cells])
    #X_q, sc = quantize_array(quant_type, serialized)
    bytes, R_limits, N_clust, nbits, npad = quantize_clustering(serialized, UInt(nbits), UInt(2))
    X_comp = transcode(CodecZlib.ZlibCompressor, reinterpret(UInt8, bytes) |> Vector{UInt8})
    cell_coords = map(cells) do cell
        cell_to_binary(cell)
    end
(;:compressed=>X_comp, :quantize_info=>(R_limits, N_clust, nbits, npad), :core_dims=>core_dims, :factor_lengths=>factor_lengths, :cell_coords=>cell_coords)
end

function unpack_cells(compressed, quantize_info, core_dims, factor_lengths, cell_coords)
    X_decomp = transcode(CodecZlib.ZlibDecompressor, compressed)
    Y = dequantize_clustering(X_decomp, quantize_info...)
    cores, factors = deserialize_leaves(Y, core_dims, factor_lengths,length(cell_coords))
    cells =  [ 
    begin
        B = cell_coords[K]
        rect = hcat(bin2f.(B)...)
        Cell( SVector{3}(rect[1,1:3]), SVector{3}(rect[2,1:3]),
            (;:D => (;:core=>cores[:,:,:,K], :factors => [factors[K][:,:,n]  for n in 1:3]), 
              :active => true) )
    end for K in axes(cores,4)]
    cells
end

function get_packed_size(packed)
  length(packed.compressed) + 
  length(packed.cell_coords)*3 +
  length(packed.quantize_info[1])*2*4 + length(packed.quantize_info[2])*4 + 1 + 1 +
  3 + # core_dims
  length(packed.factor_lengths)*2
end

end

if false
    using .VDFOctreeApprox
    N=100
    include("loadvlasi.jl")
    vdf_image = vdf_image 
    img = vdf_image3

    #A,b,img,reco, cell, tree= testapprox(size(img); fillfun=(i,j,k)->img[i,j,k], alpha=1.0, beta=1.0, nu=12, tol=1e-8);
    A,b,img,reco, cell, tree = VDFOctreeApprox.compress(img;maxiter=100, alpha=1.0, beta=1.0, nu=12, tol=1e-3)

    let treesize = length(tree)*length(b), imgsize = size(img)|>prod
        println(treesize)
        println(imgsize)
        println(treesize / imgsize)
    end
end
