import os
from secp256k1 import PrivateKey, PublicKey
import configuration


class SignKey:
    def __init__(self, hexdata=None):
        if hexdata:
            self.privkey = PrivateKey(hexdata)
        else:
            self.privkey = PrivateKey()
        self.pubkey = self.privkey.pubkey
        self.hex_pubkey = self.pubkey.serialize().hex()

    def sign(self, string):
        sig = self.privkey.ecdsa_sign(string)
        sig_der = self.privkey.ecdsa_serialize(sig).hex()
        return sig_der

    def save(self, fname):
        configuration.config["Configuration"]["derKey"] = self.privkey.serialize()
        configuration.update_config()

    def load(self, fname, create=True):
        if not configuration.config["Configuration"].get("derKey"):
            privkey = self
            privkey.save(fname)
            return privkey

        bdata = configuration.config["Configuration"]["derKey"]
        return type(self)(PrivateKey().deserialize(bdata))


class VerifyKey:
    def __init__(self, hex_pubkey):
        self.pubkey = PublicKey().deserialize(bytes(bytearray.fromhex(hex_pubkey)))
        self.pubkey = PublicKey(self.pubkey)
        self.hex_pubkey = hex_pubkey

    def verify(self, message, sign_hex):
        sig = PublicKey().ecdsa_deserialize(bytes(bytearray.fromhex(sign_hex)))
        return self.pubkey.ecdsa_verify(message, sig)


def pubkey():
    skey = SignKey().load("skey.der", create=True)
    return skey.hex_pubkey


def sign(message):
    skey = SignKey().load("skey.der", create=True)
    return skey.sign(message)


def verify(message, sig, pubkey):
    vkey = VerifyKey(pubkey)
    return vkey.verify(message, sig, pubkey)


if __name__ == "__main__":
    main()
