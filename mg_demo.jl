@inline function get2D(v, N, i, j) 
    if (j > N || j < 1 || i > N || i < 1) 
        return 0.0
    end
    return v[(i-1) * N + j]
end

function rbgs_step(xh, b, N, red=1)
    row = fld1(threadIdx().x + (blockIdx().x - 1) * blockDim().x, 32)
    if row > N 
        return
    end

    start = 1 + (mod(row, 2) != red)
    col = (mod1(threadIdx().x, 32) - 1) * 2 + start
    
    while col <= N
        xh[(row - 1) * N + col] = ((get2D(xh, N, row, col + 1) + 
                                    get2D(xh, N, row, col - 1) + 
                                    get2D(xh, N, row + 1, col) + 
                                    get2D(xh, N, row - 1, col) + 
                                    b[(row - 1) * N + col]) / 4)
        col += 32
    end

    return
end
    
function rbgs(xh, b, iters=2)
    N = isqrt(length(xh))
    threads = 256
    rows = div(threads, 32)
    blocks = cld(N, rows)
    for i=1:iters
        @cuda threads=threads blocks=blocks rbgs_step(xh, b, N, 1)
        @cuda threads=threads blocks=blocks rbgs_step(xh, b, N, 0)
    end

    return
end


function residual_gpu(xh, b, rh, N)
    row = fld1(threadIdx().x + (blockIdx().x - 1) * blockDim().x, 32)
    col = mod1(threadIdx().x, 32)
    if row > N 
        return
    end

    while col <= N
        rh[(row - 1) * N + col] = (b[(row - 1) * N + col] + 
                                    get2D(xh, N, row, col + 1) +
                                    get2D(xh, N, row, col - 1) +
                                    get2D(xh, N, row + 1, col) +
                                    get2D(xh, N, row - 1, col) -
                                    4 * xh[(row - 1) * N + col])
        col += 32
    end

    return
end

function residual(xh, b, rh)
    N = isqrt(length(xh))
    threads = 256
    rows = div(threads, 32)
    blocks = cld(N, rows)
    @cuda threads=threads blocks=blocks residual_gpu(xh, b, rh, N)
    
    return
end


function restrict_gpu(rH, rh, N, nN)
    row = fld1(threadIdx().x + (blockIdx().x - 1) * blockDim().x, 32)
    col = mod1(threadIdx().x, 32)
    if row > nN 
        return
    end

    row_h = 2 * row
    while col <= nN
        col_h = 2 * col
        rH[(row - 1) * nN + col] = 4 *
                                    (get2D(rh, N, row_h - 1, col_h - 1) / 16 +
                                    get2D(rh, N, row_h - 1, col_h) / 8 +
                                    get2D(rh, N, row_h - 1, col_h + 1) / 16 +
                                    get2D(rh, N, row_h, col_h - 1) / 8 +
                                    get2D(rh, N, row_h, col_h) / 4 +
                                    get2D(rh, N, row_h, col_h + 1) / 8 +
                                    get2D(rh, N, row_h + 1, col_h - 1) / 16 +
                                    get2D(rh, N, row_h + 1, col_h) / 8 +
                                    get2D(rh, N, row_h + 1, col_h + 1) / 16)
        col += 32
    end
    
    return
end

function restrict(rH, rh)    
    N = isqrt(length(rh))
    nN = div(N - 1, 2)
    threads = 256
    rows = div(threads, 32)
    blocks = cld(nN, rows)
    @cuda threads=threads blocks=blocks restrict_gpu(rH, rh, N, nN)
    
    return
end

function prolongation_gpu(xh, e_H, N, NH)
    row = fld1(threadIdx().x + (blockIdx().x - 1) * blockDim().x, 32)
    col = mod1(threadIdx().x, 32)
    if row > N 
        return
    end

    row_H = div(row, 2)
    while col <= N
        col_H = div(col, 2)
        result = 0
        if mod(row, 2) == 0
            if mod(col, 2) == 0
                result = get2D(e_H, NH, row_H, col_H)
            else
                result = (get2D(e_H, NH, row_H, col_H) +
                          get2D(e_H, NH, row_H, col_H + 1)) / 2
            end
        else
            if mod(col, 2) == 0
                result = (get2D(e_H, NH, row_H, col_H) +
                          get2D(e_H, NH, row_H + 1, col_H)) / 2
            else
                result = (get2D(e_H, NH, row_H, col_H) +
                          get2D(e_H, NH, row_H, col_H + 1) +
                          get2D(e_H, NH, row_H + 1, col_H) +
                          get2D(e_H, NH, row_H + 1, col_H + 1)) / 4
            end
        end
        xh[(row - 1) * N + col] += result
        col += 32
    end
    
    return
end

function prolongation(xh, e_H)    
    N = isqrt(length(xh))
    NH = div(N - 1, 2)
    threads = 256
    rows = div(threads, 32)
    blocks = cld(N, rows)
    @cuda threads=threads blocks=blocks prolongation_gpu(xh, e_H, N, NH)
    
    return
end


function mg_5pt(x, xhs, bs, ehs, N, lvls, tol=0.0001, max_iters=100)
    err = 1
    c = 0
    
    n = N*N

    while c < max_iters && err > tol
        
        for lvl=1:(lvls-1)
            xh = xhs[lvl]
            b = bs[lvl]
            rh = ehs[lvl]

            rbgs(xh, b)

            residual(xh, b, rh)

            rH = bs[lvl+1]
            restrict(rH, rh)
        end

        xhs[lvls][1] = bs[lvls][1] / 4

        for lvl=(lvls-1):-1:1
            e_H = xhs[lvl+1]
            eh = ehs[lvl]
            xh = xhs[lvl]

            prolongation(xh, e_H)

            b = bs[lvl]
            rbgs(xh, b)

            fill!(xhs[lvl+1], 0)
        end

        err = findmax(abs.(x - xhs[1]))[1]

        copyto!(x, xhs[1])
        c += 1
    end

    return c, err
end

function mg_demo(N=1023, print_results=false)
    n = N*N

    lvls = convert(Int64, log(2, N+1))

    xhs = []
    bs = []
    ehs = []

    let k = n
        for lvl=1:lvls
            push!(xhs, CuArrays.fill(0.0, k))
            push!(bs, CuArrays.fill(0.0, k))
            push!(ehs, CuArrays.fill(0.0, k))
            k = isqrt(k)
            k = div((k-1), 2)
            k = k * k
        end
    end
    
    x = CuArrays.fill(0.0, n)
    copyto!(x, xhs[1])

    b = CuArrays.fill(1.0, n)
    copyto!(bs[1], b)

    @time c, err = mg_5pt(x, xhs, bs, ehs, N, lvls)

    println(c)
    println(err)

    if print_results
        println(x)
    end
end
