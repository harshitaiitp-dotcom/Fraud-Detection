import json
import os
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
print('Connected to Blockchain:', w3.is_connected())
account = w3.eth.accounts[0]
print('Using Deployer Account:', account)
artifact_path = os.path.join(
    'artifacts',
    'contracts',
    'FraudRegistry.sol',
    'FraudRegistry.json',
)
with open(artifact_path, 'r') as f:
  contract_json = json.load(f)
abi = contract_json['abi']
bytecode = contract_json['bytecode']
FraudRegistry = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = FraudRegistry.constructor().transact({'from': account})
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
contract_address = tx_receipt.contractAddress
print(f'FraudRegistry successfully deployed at: {contract_address}')
fraud_registry = w3.eth.contract(address=contract_address, abi=abi)
target_wallet = '0x71C...392A'
risk_score = 92  
reason = 'GNN detected illicit transaction pattern & high SHAP risk score'
print('Logging fraud record on-chain...')
tx_hash = fraud_registry.functions.flagFraudulentAccount(
    target_wallet, risk_score, reason
).transact({'from': account})
w3.eth.wait_for_transaction_receipt(tx_hash)
is_blacklisted = fraud_registry.functions.checkStatus(target_wallet).call()
print(f'Wallet {target_wallet} Blacklisted on Blockchain?: {is_blacklisted}')