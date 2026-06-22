import json
import os

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

CONTRACT_ADDRESS = "0x8F9A2A0b0753d433db6Ff70f08D6B2cc62a7C45D"

w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise Exception("Failed to connect to Sepolia")

with open("blockchain/contract_abi.json", "r") as f:
    abi = json.load(f)

contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=abi
)


def store_record(video_hash, prediction):

    nonce = w3.eth.get_transaction_count(
        WALLET_ADDRESS,
        "pending"
    )

    tx = contract.functions.storeRecord(
        video_hash,
        prediction
    ).build_transaction(
        {
            "from": WALLET_ADDRESS,
            "nonce": nonce,
            "chainId": 11155111
        }
    )

    gas_estimate = w3.eth.estimate_gas(tx)

    tx["gas"] = gas_estimate

    signed_tx = w3.eth.account.sign_transaction(
        tx,
        PRIVATE_KEY
    )

    tx_hash = w3.eth.send_raw_transaction(
        signed_tx.raw_transaction
    )

    return tx_hash.hex()


def get_total_records():
    return contract.functions.getTotalRecords().call()


def get_record(index):
    return contract.functions.getRecord(index).call()