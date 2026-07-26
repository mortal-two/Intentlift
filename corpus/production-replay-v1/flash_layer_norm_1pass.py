import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0, p1, p2, p3, p4, p5, p6, p7,
    p8, p9, p10, p11, p12, p13, p14, p15,
    p16, p17, p18, p19, p20, p21,
    p22, p23, p24, p25, p26,
    p27: tl.constexpr, p28: tl.constexpr, p29: tl.constexpr,
    p30: tl.constexpr, p31: tl.constexpr, p32: tl.constexpr,
    p33: tl.constexpr, p34: tl.constexpr, p35: tl.constexpr,
    p36: tl.constexpr, c0: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, c0)
    lane_mask = col < p23

    p0_base = p0 + row * p16
    p0_lane = p0_base + col
    x0 = tl.load(p0_lane, mask=lane_mask, other=0.0).to(tl.float32)

    p5_base = p5 + row * p20
    p5_lane = p5_base + col
    x4 = tl.load(p5_lane, mask=lane_mask, other=0.0).to(tl.float32)
    summed = x0 + x4

    p4_base = p4 + row * p18
    p4_lane = p4_base + col
    x3 = tl.load(p4_lane, mask=lane_mask, other=0.0).to(tl.float32)
    summed = summed + x3

    summed_f16 = summed.to(tl.float16)
    p9_base = p9 + row * p19
    p9_lane = p9_base + col
    tl.store(p9_lane, summed_f16, mask=lane_mask)

    summed_total = tl.sum(summed, axis=0)
    extent_f32 = p23.to(tl.float32)
    mean = summed_total / extent_f32
    tl.store(p14 + row, mean)

    centered = summed - mean
    centered_or_zero = tl.where(lane_mask, centered, 0.0)
    centered_square = centered_or_zero * centered_or_zero
    square_total = tl.sum(centered_square, axis=0)
    variance = square_total / extent_f32
    variance_eps = variance + p24
    rstd = tl.rsqrt(variance_eps)
    tl.store(p15 + row, rstd)

    normalized = centered * rstd

    gamma0 = tl.load(p2 + col, mask=lane_mask).to(tl.float32)
    if p26:
        gamma0_merge = gamma0 + 1.0
    else:
        gamma0_merge = gamma0
    beta0 = tl.load(p3 + col, mask=lane_mask).to(tl.float32)
    affine0 = normalized * gamma0_merge
    affine0 = affine0 + beta0
    out0 = affine0.to(tl.float16)
    p1_base = p1 + row * p17
    p1_lane = p1_base + col
    tl.store(p1_lane, out0, mask=lane_mask)

    gamma1 = tl.load(p6 + col, mask=lane_mask).to(tl.float32)
    if p26:
        gamma1_merge = gamma1 + 1.0
    else:
        gamma1_merge = gamma1
    beta1 = tl.load(p7 + col, mask=lane_mask).to(tl.float32)
    affine1 = normalized * gamma1_merge
    affine1 = affine1 + beta1
    affine1_f16 = affine1.to(tl.float16)
    offset_f16 = p25.to(tl.float16)
    affine1_offset = (affine1_f16 + offset_f16).to(tl.float16)

    s0 = tl.where(p31, affine1_offset, affine1_f16)
    s1 = tl.where(p27, affine1_offset, s0)
    s2 = tl.where(p28, s1, affine1_offset)
    s3 = tl.where(p29, s2, affine1_offset)
    s4 = tl.where(p30, s3, affine1_offset)
    s5 = tl.where(p32, affine1_offset, s4)
    s6 = tl.where(p33, affine1_offset, s5)
    s7 = tl.where(p34, s6, affine1_offset)
    s8 = tl.where(p35, s7, affine1_offset)
    out1 = tl.where(p36, s8, affine1_offset)

    p8_base = p8 + row * p21
    p8_lane = p8_base + col
    tl.store(p8_lane, out1, mask=lane_mask)
