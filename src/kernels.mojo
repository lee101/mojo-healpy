"""HEALPix geometry kernels."""

from std.math import acos, atan2, cos, floor, sin, sqrt

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PI = 3.141592653589793238462643383279502884
comptime TWOPI = 6.283185307179586476925286766559005768


def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def interleave(ix: Int, iy: Int) -> Int:
    var x = ix
    var y = iy
    var result = 0
    var bit = 0
    while x > 0 or y > 0:
        result |= (x & 1) << (2 * bit)
        result |= (y & 1) << (2 * bit + 1)
        x >>= 1
        y >>= 1
        bit += 1
    return result


def deinterleave(code: Int, nside: Int) -> Tuple[Int, Int]:
    var ix = 0
    var iy = 0
    var scale = 1
    var bit = 0
    while scale < nside:
        ix |= ((code >> (2 * bit)) & 1) * scale
        iy |= ((code >> (2 * bit + 1)) & 1) * scale
        scale <<= 1
        bit += 1
    return (ix, iy)


def wrap_phi(phi_in: Float64) -> Float64:
    return phi_in - floor(phi_in / TWOPI) * TWOPI


def zphi_to_pix(nside: Int, z: Float64, phi_in: Float64, nest: Bool) -> Int:
    var za = abs(z)
    var tt = wrap_phi(phi_in) / (0.5 * PI)
    var jp: Int
    var jm: Int
    var face: Int
    var ix: Int
    var iy: Int

    if nest:
        if za <= 2.0 / 3.0:
            jp = Int(floor(Float64(nside) * (0.5 + tt - 0.75 * z)))
            jm = Int(floor(Float64(nside) * (0.5 + tt + 0.75 * z)))
            var ifp = jp // nside
            var ifm = jm // nside
            if ifp == ifm:
                face = ifp | 4
            elif ifp < ifm:
                face = ifp
            else:
                face = ifm + 8
            ix = jm % nside
            iy = nside - (jp % nside) - 1
        else:
            var ntt = Int(floor(tt))
            if ntt >= 4:
                ntt = 3
            var tp = tt - Float64(ntt)
            var tmp = Float64(nside) * sqrt(3.0 * (1.0 - za))
            jp = Int(floor(tp * tmp))
            jm = Int(floor((1.0 - tp) * tmp))
            if jp >= nside:
                jp = nside - 1
            if jm >= nside:
                jm = nside - 1
            if z >= 0.0:
                face = ntt
                ix = nside - jm - 1
                iy = nside - jp - 1
            else:
                face = ntt + 8
                ix = jp
                iy = jm
        return face * nside * nside + interleave(ix, iy)

    var nl4 = 4 * nside
    var ncap = 2 * nside * (nside - 1)
    var npix = 12 * nside * nside
    if za <= 2.0 / 3.0:
        var temp1 = Float64(nside) * (0.5 + tt)
        var temp2 = Float64(nside) * z * 0.75
        jp = Int(floor(temp1 - temp2))
        jm = Int(floor(temp1 + temp2))
        var ir = nside + 1 + jp - jm
        var kshift = 1 - (ir & 1)
        var ring_ip = (jp + jm - nside + kshift + 1) // 2
        ring_ip %= nl4
        if ring_ip < 0:
            ring_ip += nl4
        return ncap + (ir - 1) * nl4 + ring_ip

    var tp = tt - floor(tt)
    var tmp = Float64(nside) * sqrt(3.0 * (1.0 - za))
    jp = Int(floor(tp * tmp))
    jm = Int(floor((1.0 - tp) * tmp))
    var ir = jp + jm + 1
    var ring_ip = Int(floor(tt)) * ir + jp
    if z > 0.0:
        return 2 * ir * (ir - 1) + ring_ip
    return npix - 2 * ir * (ir + 1) + ring_ip


def ang_to_pix(nside: Int, theta: Float64, phi_in: Float64, nest: Bool) -> Int:
    return zphi_to_pix(nside, cos(theta), phi_in, nest)


def pix_to_zphi(nside: Int, pix: Int, nest: Bool) -> Tuple[Float64, Float64]:
    var z: Float64
    var phi: Float64
    if nest:
        var npface = nside * nside
        var face = pix // npface
        var code = pix % npface
        var ix, iy = deinterleave(code, nside)
        var jrll: Int
        var jpll: Int
        if face < 4:
            jrll = 2
            jpll = 2 * face + 1
        elif face < 8:
            jrll = 3
            jpll = 2 * (face - 4)
        else:
            jrll = 4
            jpll = 2 * (face - 8) + 1
        var jr = jrll * nside - ix - iy - 1
        var nr: Int
        var kshift: Int
        if jr < nside:
            nr = jr
            z = 1.0 - Float64(nr * nr) / (3.0 * Float64(nside * nside))
            kshift = 0
        elif jr > 3 * nside:
            nr = 4 * nside - jr
            z = -1.0 + Float64(nr * nr) / (3.0 * Float64(nside * nside))
            kshift = 0
        else:
            nr = nside
            z = Float64(2 * nside - jr) * (2.0 / (3.0 * Float64(nside)))
            kshift = (jr - nside) & 1
        var jp = (jpll * nr + ix - iy + 1 + kshift) // 2
        var nl4 = 4 * nr
        if jp > nl4:
            jp -= nl4
        if jp < 1:
            jp += nl4
        phi = (
            (Float64(jp) - 0.5 * Float64(kshift + 1)) * PI / (2.0 * Float64(nr))
        )
        return (z, phi)

    var npix = 12 * nside * nside
    var ncap = 2 * nside * (nside - 1)
    var nl4 = 4 * nside
    if pix < ncap:
        var iring = Int(floor(0.5 * (1.0 + sqrt(1.0 + 2.0 * Float64(pix)))))
        var iphi = pix + 1 - 2 * iring * (iring - 1)
        z = 1.0 - Float64(iring * iring) / (3.0 * Float64(nside * nside))
        phi = (Float64(iphi) - 0.5) * PI / (2.0 * Float64(iring))
    elif pix < npix - ncap:
        var shifted = pix - ncap
        var iring = shifted // nl4 + nside
        var iphi = shifted % nl4 + 1
        var fodd = 0.5 * Float64(1 + ((iring + nside) & 1))
        z = Float64(2 * nside - iring) * (2.0 / (3.0 * Float64(nside)))
        phi = (Float64(iphi) - fodd) * PI / (2.0 * Float64(nside))
    else:
        var shifted = npix - pix
        var iring = Int(floor(0.5 * (1.0 + sqrt(2.0 * Float64(shifted) - 1.0))))
        var iphi = 4 * iring + 1 - (shifted - 2 * iring * (iring - 1))
        z = -1.0 + Float64(iring * iring) / (3.0 * Float64(nside * nside))
        phi = (Float64(iphi) - 0.5) * PI / (2.0 * Float64(iring))
    return (z, phi)


def pix2ang_kernel(
    nside: Int,
    pixels_address: Int,
    theta_address: Int,
    phi_address: Int,
    n: Int,
    nest: Int,
    start: Int,
    stop: Int,
):
    # Every element costs an acos plus a divide over 24 bytes of traffic,
    # so these transforms are compute-bound and the caller splits the
    # element range across worker threads.
    var pixels = ip(pixels_address)
    var theta = fp(theta_address)
    var phi = fp(phi_address)
    for i in range(start, stop):
        var z, p = pix_to_zphi(nside, Int(pixels[i]), nest != 0)
        theta[i] = acos(max(-1.0, min(1.0, z)))
        phi[i] = p


def ang2pix_kernel(
    nside: Int,
    theta_address: Int,
    phi_address: Int,
    pixels_address: Int,
    n: Int,
    nest: Int,
    start: Int,
    stop: Int,
):
    var theta = fp(theta_address)
    var phi = fp(phi_address)
    var pixels = ip(pixels_address)
    for i in range(start, stop):
        pixels[i] = Int64(ang_to_pix(nside, theta[i], phi[i], nest != 0))


def pix2vec_kernel(
    nside: Int,
    pixels_address: Int,
    x_address: Int,
    y_address: Int,
    z_address: Int,
    n: Int,
    nest: Int,
    start: Int,
    stop: Int,
):
    var pixels = ip(pixels_address)
    var x = fp(x_address)
    var y = fp(y_address)
    var z_dst = fp(z_address)
    for i in range(start, stop):
        var z, phi = pix_to_zphi(nside, Int(pixels[i]), nest != 0)
        var r = sqrt(max(0.0, 1.0 - z * z))
        x[i] = r * cos(phi)
        y[i] = r * sin(phi)
        z_dst[i] = z


def vec2pix_kernel(
    nside: Int,
    x_address: Int,
    y_address: Int,
    z_address: Int,
    pixels_address: Int,
    n: Int,
    nest: Int,
    start: Int,
    stop: Int,
):
    var x = fp(x_address)
    var y = fp(y_address)
    var z = fp(z_address)
    var pixels = ip(pixels_address)
    for i in range(start, stop):
        var norm = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])
        var theta = acos(max(-1.0, min(1.0, z[i] / norm)))
        var phi = atan2(y[i], x[i])
        pixels[i] = Int64(ang_to_pix(nside, theta, phi, nest != 0))


def order_convert_kernel(
    nside: Int,
    source_address: Int,
    destination_address: Int,
    n: Int,
    nested_output: Int,
    start: Int,
    stop: Int,
):
    var source = ip(source_address)
    var destination = ip(destination_address)
    for i in range(start, stop):
        var z, phi = pix_to_zphi(nside, Int(source[i]), nested_output == 0)
        destination[i] = Int64(
            zphi_to_pix(nside, z, phi, nested_output != 0)
        )
