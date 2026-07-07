import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi

# Pauli matrices used for trace calculations throughout the file
X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
Z = np.array([[1, 0], [0, -1]], dtype=DTYPE)


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    b = Bloch()
    
    # Get the global phase from the determinant (divide by 2)
    det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    b.alpha = np.angle(det) / 2.0

    # Remove the global phase to isolate the pure SU(2) rotation
    u = g * np.exp(-1j * b.alpha)

    # Extract the cosine component using the trace
    cos_half = np.real(np.trace(u)) / 2.0
    
    # Extract the scaled axis components using the Pauli matrices
    nx_sin = np.real(1j * np.trace(X @ u)) / 2.0
    ny_sin = np.real(1j * np.trace(Y @ u)) / 2.0
    nz_sin = np.real(1j * np.trace(Z @ u)) / 2.0
    
    # Find the sine component by taking the magnitude of the scaled vector
    sin_half = np.linalg.norm([nx_sin, ny_sin, nz_sin])
    
    # Safely calculate the full angle using arctan2
    b.theta = 2.0 * np.arctan2(sin_half, cos_half)

    # Extract the normalized axis (handling the zero-rotation edge case)
    if sin_half < 1e-10:
        b.n = np.array([0.0, 0.0, 1.0])
    else:
        b.n = np.array([nx_sin, ny_sin, nz_sin]) / sin_half
            
    return b


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)

# We define standard unit vectors for x, y, and z directions
x_hat = np.array([1.0, 0.0, 0.0])
y_hat = np.array([0.0, 1.0, 0.0])
z_hat = np.array([0.0, 0.0, 1.0])

# Calculate the cotangent of pi/8
cot_pi8 = 1.0 / np.tan(np.pi / 8.0)

# Apply the mathematical simplifications for the raw axes
n1_unnorm = cot_pi8 * (z_hat - x_hat) + y_hat
n2_unnorm = np.sqrt(2.0) * cot_pi8 * y_hat - (z_hat - x_hat) / np.sqrt(2.0)

# Divide them by their length (norm) so they have a length of exactly 1
n1 = n1_unnorm / np.linalg.norm(n1_unnorm)
n2 = n2_unnorm / np.linalg.norm(n2_unnorm)

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    # Calculate the cosine and sine of the half-angle
    c = np.cos(b.theta / 2.0)
    s = np.sin(b.theta / 2.0)
    
    # Scale the rotation axis 'n' by the sine of the half angle
    v = s * b.n
    
    # Project the vector onto our orthogonal axes
    v1 = np.dot(v, a1)
    v2 = np.dot(v, a2)
    v3 = np.dot(v, a3)
    
    # Solve the system of trigonometric equations.
    sum_angles = np.arctan2(v1, c)   
    diff_angles = np.arctan2(v3, v2) 
    
    # Extract the FULL alpha and gamma angles algebraically
    alpha = sum_angles - diff_angles
    gamma = sum_angles + diff_angles
    
    # Extract the half-angle for beta, then multiply by 2 for the FULL angle
    beta_half = np.arctan2(np.sqrt(v2**2 + v3**2), np.sqrt(c**2 + v1**2))
    beta = 2.0 * beta_half
    
    return alpha, beta, gamma, b.alpha


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    target = angle % TWO_PI
    k = 1
    
    while True:
        # Calculate the current multiple's angle and wrap it within [0, 2*pi)
        current_angle = (k * LAMBDA_PI) % TWO_PI
        difference = abs(current_angle - target)
        
        # Calculate the true angular distance accounting for the 2*pi wrap-around
        distance = min(difference, TWO_PI - difference)
        
        if distance <= tolerance:
            return k
            
        k += 1


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """
    b = to_bloch(u)
    alpha, beta, gamma, _global_phase = n1n2n1_angles(b)
    
    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)
    
    return k, l, m


# ---------------------------------------------------------------------------
# Single-qubit rotation helpers (see cpp/src/Unitary2_Bloch.h).
#
# These are the inverse/companion operations to to_bloch and are reused by the
# multi-qubit decomposition pipeline in decompose.py.
# ---------------------------------------------------------------------------


def from_axis_angle(b: Bloch) -> np.ndarray:
    """Build a 2x2 unitary from its Bloch form: a global phase times a rotation
    by angle b.theta about axis b.n (inverse of to_bloch).

        G = e^{i b.alpha} (cos(b.theta/2) I - i sin(b.theta/2) (b.n . sigma))

    where (b.n . sigma) = n_x X + n_y Y + n_z Z. Assumes b.n is a unit vector.
    """
    I = np.eye(2, dtype=DTYPE)
    n_dot_sigma = b.n[0] * X + b.n[1] * Y + b.n[2] * Z
    return np.exp(1j * b.alpha) * (np.cos(b.theta / 2.0) * I - 1j * np.sin(b.theta / 2.0) * n_dot_sigma)


def Rz(theta: float) -> np.ndarray:
    """Rotation about the z axis (no global phase):

    Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}).
    """
    b = Bloch()
    b.alpha = 0.0
    b.n = np.array([0.0, 0.0, 1.0])
    b.theta = theta
    return from_axis_angle(b)


def Ry(theta: float) -> np.ndarray:
    """Rotation about the y axis (no global phase):

    Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]].
    """
    b = Bloch()
    b.alpha = 0.0
    b.n = np.array([0.0, 1.0, 0.0])
    b.theta = theta
    return from_axis_angle(b)


def euler_angles_zyz(u: np.ndarray) -> tuple[float, float, float, float]:
    """ZYZ Euler decomposition of a 2x2 unitary: angles (alpha, beta, gamma, delta)
    with

        u = e^{i alpha} Rz(beta) Ry(gamma) Rz(delta).

    alpha is the global phase (arg(det u)/2); the rest come from S = e^{-i alpha} u
    in SU(2), where s00 = cos(gamma/2) e^{-i(beta+delta)/2} and
    s10 = sin(gamma/2) e^{i(beta-delta)/2}. When gamma = 0 (s10 = 0), beta/delta are
    split arbitrarily (gimbal lock); the identity still holds.
    """
    alpha = np.angle(np.linalg.det(u)) / 2.0
    S = np.exp(-1j * alpha) * u
    
    s00 = S[0, 0]
    s10 = S[1, 0]
    
    # Extract gamma handling floating point inaccuracies
    gamma = 2.0 * np.arcsin(np.clip(np.abs(s10), 0.0, 1.0))
    
    # Handle gimbal lock case (gamma = 0)
    if np.isclose(np.abs(s10), 0.0, atol=1e-10):
        beta = -2.0 * np.angle(s00)
        delta = 0.0
    else:
        beta_plus_delta = -2.0 * np.angle(s00)
        beta_minus_delta = 2.0 * np.angle(s10)
        beta = (beta_plus_delta + beta_minus_delta) / 2.0
        delta = (beta_plus_delta - beta_minus_delta) / 2.0
        
    return alpha, beta, gamma, delta


def unitary2_sqrt(u: np.ndarray) -> np.ndarray:
    """Principal square root: a 2x2 unitary V with V @ V == u, phase included.
    Take the Bloch form of u and halve both alpha and theta (same axis); squaring
    back doubles them, reproducing u exactly.
    """
    b = to_bloch(u)
    b.alpha /= 2.0
    b.theta /= 2.0
    return from_axis_angle(b)


# ---------------------------------------------------------------------------
# H/T word machinery for approximating a 2x2 unitary in {H, T} (see cpp/src/HT.h).
#
# M1, M2 are short H/T words that realize rotations by THETA_M = 2*LAMBDA_PI about
# the axes a1, a2. A word is a flat string of 'H'/'T' characters, read left-to-right
# as a matrix product (leftmost char = leftmost/outermost factor).
# ---------------------------------------------------------------------------

# alternating (T-power, H-power, ...) exponents, starting with T
M1_WORD = [7, 1, 1, 1]
M2_WORD = [2, 1, 1, 1, 6, 1, 7, 1, 5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 7, 1, 6]


def expand_word(word: list[int]) -> str:
    """Flatten an alternating (T-power, H-power, ...) exponent list into a literal
    string of 'H'/'T' gates (left-to-right). Even indices are T, odd indices are H.
    """
    res = ""
    for i, power in enumerate(word):
        char = 'T' if i % 2 == 0 else 'H'
        res += char * power
    return res


# flat H/T strings for the two building-block words (computed once expand_word works)
M1_STR = expand_word(M1_WORD)
M2_STR = expand_word(M2_WORD)


def gates_to_unitary(gates: str) -> np.ndarray:
    """The 2x2 unitary of a flat H/T gate string (left-to-right product)."""
    res = np.eye(2, dtype=DTYPE)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=DTYPE)
    
    for char in gates:
        if char == 'H':
            res = res @ H
        elif char == 'T':
            res = res @ T
    return res


def invert_gates(gates: str) -> str:
    """Inverse of a flat H/T word: reverse the gate order and invert each gate.
    H^-1 = H; the {H, T} basis has no T-dagger, so T^-1 must be spelled as T^7.
    """
    res = ""
    for char in reversed(gates):
        if char == 'H':
            res += 'H'
        elif char == 'T':
            res += 'T' * 7  # T^-1 is T^7
    return res


def power_gates(base: str, k: int) -> str:
    """The k-th power of a flat H/T word: base repeated k times. Negative k uses the
    inverse word (invert_gates).
    """
    if k >= 0:
        return base * k
    return invert_gates(base) * abs(k)


def approximate_in_ht(u: np.ndarray, error: float) -> str:
    """Approximate a 2x2 unitary `u` by a flat H/T word (up to global phase) to the
    angular tolerance `error` (smaller -> longer, more accurate).

    Use decompose_2x2 to get the powers (k, l, m) with u ~= M1^k M2^l M1^m, then
    assemble the word:

        power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m).
    """
    k, l, m = decompose_2x2(u, error)
    return power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m)
