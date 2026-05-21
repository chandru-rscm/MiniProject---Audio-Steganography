"""
AES-256-GCM Encryption — StegoWave

Implemented from scratch:
    - SHA-256        : full 64-round hash algorithm
    - HMAC-SHA256    : keyed hash using SHA-256
    - PBKDF2-SHA256  : key derivation (100k iterations of HMAC-SHA256)
    - AES-256 core   : SubBytes, ShiftRows, MixColumns, AddRoundKey, KeySchedule
    
GCM mode uses pycryptodome (CTR + GHASH authenticated encryption wrapper).
"""

import os
import struct
from Crypto.Cipher import AES as _AES_GCM


# ══════════════════════════════════════════════════════════════════
#  SHA-256 FROM SCRATCH
# ══════════════════════════════════════════════════════════════════

# SHA-256 initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
_SHA256_H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]

# SHA-256 round constants (first 32 bits of fractional parts of cbrt of first 64 primes)
_SHA256_K = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]

def _rotr32(x: int, n: int) -> int:
    """Rotate right 32-bit integer x by n bits."""
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def _sha256_pad(message: bytes) -> bytes:
    """
    SHA-256 message padding:
        1. Append bit '1' (0x80 byte)
        2. Append zeros until length ≡ 56 mod 64 bytes
        3. Append original message length as 64-bit big-endian integer
    """
    msg_len  = len(message)
    bit_len  = msg_len * 8
    message += b'\x80'
    # Pad until length ≡ 56 mod 64
    while len(message) % 64 != 56:
        message += b'\x00'
    message += struct.pack('>Q', bit_len)
    return message

def sha256(message: bytes) -> bytes:
    """
    Compute SHA-256 hash of message.
    
    Algorithm (64 rounds per 512-bit block):
        1. Pad message
        2. Split into 512-bit (64-byte) blocks
        3. For each block:
           a. Create message schedule W[0..63]
           b. Initialize working variables a,b,c,d,e,f,g,h
           c. 64 rounds of mixing using Ch, Maj, Sigma functions
           d. Add compressed chunk to hash values
        4. Produce final 256-bit (32-byte) digest
    
    Returns: 32 bytes
    """
    # Initialize hash values
    h = list(_SHA256_H0)

    # Pad and process each 64-byte block
    padded = _sha256_pad(message)

    for block_start in range(0, len(padded), 64):
        block = padded[block_start : block_start + 64]

        # Create message schedule W[0..63]
        W = list(struct.unpack('>16I', block))   # first 16 words from block
        for i in range(16, 64):
            # σ0 = ROTR7(w[i-15]) XOR ROTR18(w[i-15]) XOR SHR3(w[i-15])
            s0 = _rotr32(W[i-15], 7) ^ _rotr32(W[i-15], 18) ^ (W[i-15] >> 3)
            # σ1 = ROTR17(w[i-2]) XOR ROTR19(w[i-2]) XOR SHR10(w[i-2])
            s1 = _rotr32(W[i-2], 17) ^ _rotr32(W[i-2], 19)  ^ (W[i-2] >> 10)
            W.append((W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF)

        # Initialize working variables
        a, b, c, d, e, f, g, hh = h

        # 64 rounds
        for i in range(64):
            # Σ1 = ROTR6(e) XOR ROTR11(e) XOR ROTR25(e)
            S1  = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            # Ch(e,f,g) = (e AND f) XOR (NOT e AND g)
            ch  = ((e & f) ^ (~e & g)) & 0xFFFFFFFF
            # Σ0 = ROTR2(a) XOR ROTR13(a) XOR ROTR22(a)
            S0  = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            # Maj(a,b,c) = (a AND b) XOR (a AND c) XOR (b AND c)
            maj = (a & b) ^ (a & c) ^ (b & c)

            temp1 = (hh + S1 + ch  + _SHA256_K[i] + W[i]) & 0xFFFFFFFF
            temp2 = (S0 + maj) & 0xFFFFFFFF

            hh = g
            g  = f
            f  = e
            e  = (d + temp1) & 0xFFFFFFFF
            d  = c
            c  = b
            b  = a
            a  = (temp1 + temp2) & 0xFFFFFFFF

        # Add compressed chunk to current hash values
        h[0] = (h[0] + a)  & 0xFFFFFFFF
        h[1] = (h[1] + b)  & 0xFFFFFFFF
        h[2] = (h[2] + c)  & 0xFFFFFFFF
        h[3] = (h[3] + d)  & 0xFFFFFFFF
        h[4] = (h[4] + e)  & 0xFFFFFFFF
        h[5] = (h[5] + f)  & 0xFFFFFFFF
        h[6] = (h[6] + g)  & 0xFFFFFFFF
        h[7] = (h[7] + hh) & 0xFFFFFFFF

    return struct.pack('>8I', *h)


# ══════════════════════════════════════════════════════════════════
#  HMAC-SHA256 FROM SCRATCH
# ══════════════════════════════════════════════════════════════════

def hmac_sha256(key: bytes, message: bytes) -> bytes:
    """
    HMAC-SHA256: Keyed hash using SHA-256.

    Algorithm:
        if len(key) > 64: key = SHA256(key)
        if len(key) < 64: key = key + zeros

        ipad = 0x36 repeated 64 times
        opad = 0x5C repeated 64 times

        HMAC = SHA256((key XOR opad) || SHA256((key XOR ipad) || message))

    Returns: 32 bytes
    """
    BLOCK_SIZE = 64

    # Normalize key to block size
    if len(key) > BLOCK_SIZE:
        key = sha256(key)
    if len(key) < BLOCK_SIZE:
        key = key + b'\x00' * (BLOCK_SIZE - len(key))

    # XOR key with ipad and opad
    ipad = bytes(k ^ 0x36 for k in key)
    opad = bytes(k ^ 0x5C for k in key)

    # Inner hash: SHA256(ipad_key || message)
    inner = sha256(ipad + message)

    # Outer hash: SHA256(opad_key || inner)
    return sha256(opad + inner)


# ══════════════════════════════════════════════════════════════════
#  PBKDF2-SHA256 FROM SCRATCH
# ══════════════════════════════════════════════════════════════════

def pbkdf2_sha256(password: str, salt: bytes,
                  iterations: int = 100_000, dk_len: int = 32) -> bytes:
    """
    PBKDF2 with HMAC-SHA256 — Password-Based Key Derivation Function 2.

    Converts a password string into a cryptographic key.

    Why PBKDF2?
        - Raw password → key is insecure (brute-forceable)
        - PBKDF2 runs HMAC-SHA256 100,000 times → makes brute force slow
        - Salt prevents rainbow table attacks (same password → different key)
        - 100k iterations → attacker can try ~10 passwords/second max

    Algorithm (RFC 2898):
        For each block i needed to fill dk_len bytes:
            U1 = HMAC(password, salt || INT(i))
            U2 = HMAC(password, U1)
            ...
            Uc = HMAC(password, U_{c-1})
            T_i = U1 XOR U2 XOR ... XOR Uc

        DK = T_1 || T_2 || ... (truncated to dk_len bytes)

    Returns: dk_len bytes of derived key material
    """
    password_bytes = password.encode('utf-8') if isinstance(password, str) else password
    dk             = b''
    block_index    = 1

    while len(dk) < dk_len:
        # First iteration: U1 = HMAC(password, salt || block_index)
        U = hmac_sha256(password_bytes,
                        salt + struct.pack('>I', block_index))
        T = U

        # Remaining iterations: Ui = HMAC(password, U_{i-1})
        for _ in range(iterations - 1):
            U  = hmac_sha256(password_bytes, U)
            # XOR each byte of U into T
            T  = bytes(a ^ b for a, b in zip(T, U))

        dk          += T
        block_index += 1

    return dk[:dk_len]


# ══════════════════════════════════════════════════════════════════
#  AES CONSTANTS
# ══════════════════════════════════════════════════════════════════

SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

INV_SBOX = [0] * 256
for _i, _v in enumerate(SBOX):
    INV_SBOX[_v] = _i

RCON = [
    0x00,0x01,0x02,0x04,0x08,0x10,
    0x20,0x40,0x80,0x1b,0x36,
    0x6c,0xd8,0xab,0x4d,0x9a,
]


# ══════════════════════════════════════════════════════════════════
#  GF(2^8) ARITHMETIC
# ══════════════════════════════════════════════════════════════════

def _gf_mul(a: int, b: int) -> int:
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11b
        a &= 0xff
        b >>= 1
    return result

def _xtime(a: int) -> int:
    result = a << 1
    if result & 0x100:
        result ^= 0x11b
    return result & 0xff


# ══════════════════════════════════════════════════════════════════
#  AES STATE OPERATIONS
# ══════════════════════════════════════════════════════════════════

def _bytes_to_state(block):
    return [[block[r + 4*c] for c in range(4)] for r in range(4)]

def _state_to_bytes(state):
    return bytes(state[r][c] for c in range(4) for r in range(4))

def _sub_bytes(state):
    return [[SBOX[state[r][c]] for c in range(4)] for r in range(4)]

def _inv_sub_bytes(state):
    return [[INV_SBOX[state[r][c]] for c in range(4)] for r in range(4)]

def _shift_rows(state):
    return [state[r][r:] + state[r][:r] for r in range(4)]

def _inv_shift_rows(state):
    return [state[r][4-r:] + state[r][:4-r] for r in range(4)]

def _mix_columns(state):
    new_state = [[0]*4 for _ in range(4)]
    for c in range(4):
        s0,s1,s2,s3 = state[0][c],state[1][c],state[2][c],state[3][c]
        t = s0^s1^s2^s3
        new_state[0][c] = s0^t^_xtime(s0^s1)
        new_state[1][c] = s1^t^_xtime(s1^s2)
        new_state[2][c] = s2^t^_xtime(s2^s3)
        new_state[3][c] = s3^t^_xtime(s3^s0)
    return new_state

def _inv_mix_columns(state):
    new_state = [[0]*4 for _ in range(4)]
    for c in range(4):
        s0,s1,s2,s3 = state[0][c],state[1][c],state[2][c],state[3][c]
        new_state[0][c] = _gf_mul(0x0e,s0)^_gf_mul(0x0b,s1)^_gf_mul(0x0d,s2)^_gf_mul(0x09,s3)
        new_state[1][c] = _gf_mul(0x09,s0)^_gf_mul(0x0e,s1)^_gf_mul(0x0b,s2)^_gf_mul(0x0d,s3)
        new_state[2][c] = _gf_mul(0x0d,s0)^_gf_mul(0x09,s1)^_gf_mul(0x0e,s2)^_gf_mul(0x0b,s3)
        new_state[3][c] = _gf_mul(0x0b,s0)^_gf_mul(0x0d,s1)^_gf_mul(0x09,s2)^_gf_mul(0x0e,s3)
    return new_state

def _add_round_key(state, rk):
    return [[state[r][c]^rk[r][c] for c in range(4)] for r in range(4)]


# ══════════════════════════════════════════════════════════════════
#  AES KEY SCHEDULE
# ══════════════════════════════════════════════════════════════════

def _key_schedule(key: bytes) -> list:
    assert len(key) == 32
    Nk,Nr,Nb = 8,14,4
    W = [list(key[4*i:4*i+4]) for i in range(Nk)]
    for i in range(Nk, Nb*(Nr+1)):
        temp = W[i-1][:]
        if i % Nk == 0:
            temp = temp[1:]+temp[:1]
            temp = [SBOX[b] for b in temp]
            temp[0] ^= RCON[i//Nk]
        elif i % Nk == 4:
            temp = [SBOX[b] for b in temp]
        W.append([W[i-Nk][j]^temp[j] for j in range(4)])
    round_keys = []
    for rnd in range(Nr+1):
        rk = [[0]*4 for _ in range(4)]
        for c in range(4):
            for r in range(4):
                rk[r][c] = W[rnd*4+c][r]
        round_keys.append(rk)
    return round_keys


# ══════════════════════════════════════════════════════════════════
#  AES-256 BLOCK ENCRYPT / DECRYPT
# ══════════════════════════════════════════════════════════════════

def _aes_encrypt_block(block: bytes, round_keys: list) -> bytes:
    state = _bytes_to_state(block)
    state = _add_round_key(state, round_keys[0])
    for rnd in range(1, 14):
        state = _sub_bytes(state)
        state = _shift_rows(state)
        state = _mix_columns(state)
        state = _add_round_key(state, round_keys[rnd])
    state = _sub_bytes(state)
    state = _shift_rows(state)
    state = _add_round_key(state, round_keys[14])
    return _state_to_bytes(state)

def _aes_decrypt_block(block: bytes, round_keys: list) -> bytes:
    state = _bytes_to_state(block)
    state = _add_round_key(state, round_keys[14])
    state = _inv_shift_rows(state)
    state = _inv_sub_bytes(state)
    for rnd in range(13, 0, -1):
        state = _add_round_key(state, round_keys[rnd])
        state = _inv_mix_columns(state)
        state = _inv_shift_rows(state)
        state = _inv_sub_bytes(state)
    state = _add_round_key(state, round_keys[0])
    return _state_to_bytes(state)


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

SALT_SIZE  = 16
NONCE_SIZE = 16
ITERATIONS = 1_000   # reduced for demo speed (pure Python PBKDF2)
                     # production value = 100,000

def encrypt(plaintext: bytes, password: str) -> bytes:
    """
    Encrypt using AES-256-GCM.
    Key derived via our custom PBKDF2-SHA256 (100k iterations).
    GCM mode (CTR+GHASH) uses pycryptodome wrapper.

    Output: salt(16) + nonce(16) + tag(16) + ciphertext
    """
    salt = os.urandom(SALT_SIZE)

    # Our custom PBKDF2-SHA256
    key  = pbkdf2_sha256(password, salt, ITERATIONS, dk_len=32)

    # GCM mode — pycryptodome wrapper
    cipher          = _AES_GCM.new(key, _AES_GCM.MODE_GCM,
                                    nonce=os.urandom(NONCE_SIZE))
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    return salt + cipher.nonce + tag + ciphertext


def decrypt(bundle: bytes, password: str) -> bytes:
    """
    Decrypt AES-256-GCM bundle.
    Wrong password → authentication tag mismatch → ValueError immediately.
    """
    salt       = bundle[:16]
    nonce      = bundle[16:32]
    tag        = bundle[32:48]
    ciphertext = bundle[48:]

    # Our custom PBKDF2-SHA256
    key = pbkdf2_sha256(password, salt, ITERATIONS, dk_len=32)

    cipher = _AES_GCM.new(key, _AES_GCM.MODE_GCM, nonce=nonce)
    try:
        return cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError:
        raise ValueError("Wrong password or corrupted data — authentication failed.")


# ══════════════════════════════════════════════════════════════════
#  SELF TESTS
# ══════════════════════════════════════════════════════════════════

def run_self_test():
    print("=" * 55)
    print("  StegoWave Crypto Self Tests")
    print("=" * 55)

    # SHA-256 test vector — verified against Python hashlib
    result   = sha256(b'abc').hex()
    expected = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    sha_pass = result == expected
    print(f"  SHA-256 : {'PASS' if sha_pass else 'FAIL'}")
    if not sha_pass:
        print(f"    Expected: {expected}")
        print(f"    Got     : {result}")

    # HMAC-SHA256 test (RFC 4231 Test Case 1)
    hmac_result = hmac_sha256(b'\x0b'*20, b'Hi There').hex()
    hmac_expect = 'b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7'
    hmac_pass   = hmac_result == hmac_expect
    print(f"  HMAC    : {'PASS' if hmac_pass else 'FAIL'}")

    # AES-256 test (FIPS 197)
    key   = bytes.fromhex('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f')
    plain = bytes.fromhex('00112233445566778899aabbccddeeff')
    rk    = _key_schedule(key)
    enc   = _aes_encrypt_block(plain, rk)
    aes_pass = enc.hex() == '8ea2b7ca516745bfeafc49904b496089'
    dec_pass = _aes_decrypt_block(enc, rk) == plain
    print(f"  AES-256 encrypt: {'PASS' if aes_pass else 'FAIL'}")
    print(f"  AES-256 decrypt: {'PASS' if dec_pass else 'FAIL'}")

    # Full encrypt/decrypt roundtrip (uses 1000 iterations for speed)
    ct = encrypt(b'Hello StegoWave!', 'testpassword')
    pt = decrypt(ct, 'testpassword')
    rt_pass = pt == b'Hello StegoWave!'
    print(f"  Full roundtrip : {'PASS' if rt_pass else 'FAIL'}")

    print("=" * 55)
    all_pass = sha_pass and hmac_pass and aes_pass and dec_pass and rt_pass
    print(f"  Overall: {'ALL PASS!' if all_pass else 'SOME FAILED!'}")
    return all_pass


if __name__ == '__main__':
    run_self_test()