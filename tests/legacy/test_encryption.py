# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json

from dashscope.api_entities.encryption import Encryption


class TestEncryption:
    @staticmethod
    def test_get_public_keys():
        pub_keys = Encryption._get_public_keys()
        print(f"\nrsa:\n{json.dumps(pub_keys, indent=4, ensure_ascii=False)}")
        print(f"\npublic_key_id: {pub_keys.get('public_key_id')}")
        print(f"\npublic_key: {pub_keys.get('public_key')}")

    @staticmethod
    def test_generate_aes_secret_key(self):
        key = Encryption._generate_aes_secret_key()
        print(f"\nkey: {key}")

    @staticmethod
    def test_generate_aes_iv(self):
        iv = Encryption._generate_iv()
        print(f"\niv: {iv}")

    @staticmethod
    def test_encrypt_with_aes():
        key = Encryption._generate_aes_secret_key()
        iv = Encryption._generate_iv()
        text = "hello world"
        ciphertext = Encryption._encrypt_text_with_aes(text, key, iv)
        print(f"\nciphertext: {ciphertext}")

    @staticmethod
    def test_encrypt_aes_key_with_rsa():
        public_keys = Encryption._get_public_keys()
        public_key = public_keys.get("public_key")
        aes_key = Encryption._generate_aes_secret_key()

        cipher_aes_key = Encryption._encrypt_aes_key_with_rsa(
            aes_key,
            public_key,
        )
        print(f"\ncipher_aes_key: {cipher_aes_key}")
