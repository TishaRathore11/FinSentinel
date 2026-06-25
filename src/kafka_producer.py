from kafka import KafkaProducer
import json
import time
import random
import numpy as np

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)


def generate_transaction():
    txn = {
        'Amount_log': float(np.random.uniform(0, 10)),
        'Hour': float(random.randint(0, 23)),
        'anomaly_score': 0.0
    }
    for i in range(1, 29):
        txn[f'V{i}'] = float(np.random.normal(0, 1))
    return txn


if __name__ == "__main__":
    while True:
        txn = generate_transaction()
        producer.send('transactions', value=txn)
        time.sleep(0.5)