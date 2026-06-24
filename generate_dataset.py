import random
import pandas as pd

categories = {
    "shopping": [
    "bought groceries at supermarket",
    "amazon order for electronics",
    "purchased clothes at mall",
    "ordered shoes online",
    "bought dress at mall",
    "bought jeans",
    "shopping for clothes",
    "bought t shirt",
    "purchased jacket",
    "clothing store purchase",
    "dress purchase",
    "buying clothes"
],
    
    "food": [
        "pizza order from dominos",
        "burger from mcdonalds",
        "food delivery from swiggy",
        "zomato food order",
        "coffee at cafe"
    ],
    "transport": [
        "uber ride payment",
        "ola cab ride",
        "metro ticket purchase",
        "bus ticket payment",
        "train ticket booking"
    ],
    "bills": [
        "electricity bill payment",
        "water bill payment",
        "mobile recharge",
        "internet bill payment",
        "gas bill payment"
    ],
    "entertainment": [
        "netflix subscription payment",
        "spotify premium payment",
        "movie ticket booking",
        "amazon prime subscription",
        "youtube premium payment"
    ],
    "salary": [
        "monthly salary credited",
        "salary bank transfer",
        "company salary payment"
    ]
}

rows = []

for i in range(1000):
    category = random.choice(list(categories.keys()))
    text = random.choice(categories[category])
    rows.append([text, category])

df = pd.DataFrame(rows, columns=["text", "category"])

df.to_csv("Data/transactions_1000_clean.csv", index=False)

print("Dataset created successfully!")