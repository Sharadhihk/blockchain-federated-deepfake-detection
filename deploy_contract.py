from web3 import Web3
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

# Connect to Sepolia
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise Exception("Failed to connect to Sepolia")

print("Connected to Sepolia")

# Read bytecode
with open("blockchain/contract_bytecode.txt", "r") as f:
    bytecode = f.read().strip()

# Read ABI
with open("blockchain/contract_abi.json", "r") as f:
    abi = f.read()

import json
abi = json.loads(abi)

# Contract object
Contract = w3.eth.contract(
    abi=abi,
    bytecode=bytecode
)

# Get nonce
nonce = w3.eth.get_transaction_count(WALLET_ADDRESS)

# Build deployment transaction
transaction = Contract.constructor().build_transaction(
    {
        "from": WALLET_ADDRESS,
        "nonce": nonce,
        "gas": 3000000,
        "gasPrice": w3.eth.gas_price,
        "chainId": 11155111
    }
)

# Sign transaction
signed_txn = w3.eth.account.sign_transaction(
    transaction,
    private_key=PRIVATE_KEY
)

# Send transaction
tx_hash = w3.eth.send_raw_transaction(
    signed_txn.raw_transaction
)

print(f"Transaction Hash: {tx_hash.hex()}")

# Wait for receipt
receipt = w3.eth.wait_for_transaction_receipt(
    tx_hash
)

print(f"Contract Address: {receipt.contractAddress}")