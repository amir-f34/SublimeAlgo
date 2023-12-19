from secp256k1 import PrivateKey, PublicKey


class SignKey:
    def __init__(self, der=None):
        if der:
            self.privkey = PrivateKey.deserialize(der)
        else:
            self.privkey = PrivateKey()
        self.pubkey = self.privkey.pubkey
        self.hex_pubkey = self.pubkey.serialize().hex()

    def sign(self, string):
        sig = self.privkey.ecdsa_sign(string)
        sig_der = self.privkey.ecdsa_serialize(sig).hex()
        return sig_der

    def save(self, fname):
        open(fname, "w").write(self.privkey.serialize().hex())

    def load(self, fname):
        bdata = bytes(bytearray.fromhex(open(fname, "r").read()))
        return self(der=PrivateKey().deserialize(bdata))


class VerifyKey:
    def __init__(self, hex_pubkey):
        self.pubkey = PublicKey().deserialize(bytes(bytearray.fromhex(hex_pubkey)))
        self.pubkey = PublicKey(self.pubkey)
        self.hex_pubkey = hex_pubkey

    def verify(self, message, sign_hex):
        sig = PublicKey().ecdsa_deserialize(bytes(bytearray.fromhex(sign_hex)))
        return self.pubkey.ecdsa_verify(message, sig)


def pubkey():
    skey = SignKey().load("skey.der")
    return skey.hex_pubkey


def sign(message):
    skey = SignKey().load("skey.der")
    return skey.sign(message)


def verify(message, sig, pubkey):
    vkey = VerifyKey(pubkey)
    return vkey.verify(message, sig)


if __name__ == "__main__":
    main()
