from kafka import KafkaConsumer
import json
import requests

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    txn = message.value
    try:
        response = requests.post("http://localhost:8000/predict", json=txn)
        result   = response.json()
        print(f"fraud={result['is_fraud']} | prob={result['fraud_probability']} | risk={result['risk_level']}")
    except Exception as e:
        print(f"Error: {e}")