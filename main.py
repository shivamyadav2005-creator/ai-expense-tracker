import pandas as pd
import re
import csv
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("Data/transactions_1000_clean.csv")

# Training data
X = data["text"]
y = data["category"]

# Convert text to numbers
vectorizer = CountVectorizer(ngram_range=(1,2))
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

print("Model trained successfully!\n")


# Function to extract amount
def extract_amount(text):
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return 0


# NEW FUNCTION: split multiple transactions
def split_transactions(text):

    matches = re.findall(r'[a-zA-Z\s]+?\s\d+', text)

    transactions = []

    for m in matches:
        transactions.append(m.strip())

    if len(transactions) == 0:
        transactions.append(text)

    return transactions


# NEW FUNCTION: CLI Dashboard (Formatting Improved Only)
def display_dashboard(expenses):

    if len(expenses) == 0:
        print("\nNo expenses to display.\n")
        return

    print("\n=========== CLI EXPENSE DASHBOARD ===========\n")

    max_value = max(expenses.values())
    total = sum(expenses.values())

    # HEADER
    print(f"{'CATEGORY':15} | {'SPENDING BAR':20} | {'AMOUNT':>6} | {'%':>6}")
    print("-" * 60)

    for cat, val in expenses.items():

        bar_length = int((val / max_value) * 20)
        bar = "█" * bar_length

        percent = (val / total) * 100

        print(f"{cat.upper():15} | {bar:<20} | {val:>6} | {percent:>5.1f}%")

    print("\n================================================\n")


# Budget limit
budget_limit = 5000

# Recommended category percentages
category_limits = {
    "food": 0.30,
    "transport": 0.20,
    "entertainment": 0.20,
    "shopping": 0.30
}

# Store expenses summary
expenses = {}

# History file
history_file = "history.csv"


# Main program loop
while True:

    user_input = input("Enter transaction (or type 'exit', 'graph', 'report', 'history', 'dashboard'): ")

    command = user_input.lower().strip()

    # Exit program
    if command == "exit":
        print("Program closed.")
        break


    # GRAPH
    elif command == "graph":

        if len(expenses) == 0:
            print("No expenses yet.\n")
            continue

        categories = list(expenses.keys())
        values = list(expenses.values())

        plt.pie(values, labels=categories, autopct='%1.1f%%')
        plt.title("Expense Distribution")
        plt.show()

        continue


    # HISTORY
    elif command == "history":

        if not os.path.exists(history_file):
            print("\nNo transaction history found.\n")
            continue

        df = pd.read_csv(history_file)

        if df.empty:
            print("\nNo transaction history found.\n")
        else:
            print("\n------ Transaction History ------\n")
            print(df.to_string(index=False))
            print("\n---------------------------------\n")

        continue


    # DASHBOARD COMMAND
    elif command == "dashboard":
        display_dashboard(expenses)
        continue


    # REPORT
    elif command == "report":

        if len(expenses) == 0:
            print("No expenses yet.\n")
            continue

        total = sum(expenses.values())

        print("\n------ Expense Report ------")
        print("Total Spending:", total)

        highest = max(expenses, key=expenses.get)
        print("Highest Spending Category:", highest)

        print("\nCategory Breakdown:")

        for cat, val in expenses.items():
            percent = (val / total) * 100
            print(cat, ":", val, f"({percent:.2f}%)")

        # Budget status
        if total > budget_limit:
            print("\n⚠ Warning: High spending this month!")
        else:
            print("\nSpending is under control.")

        # CATEGORY SUGGESTIONS
        print("\nRecommended Category Limits:")

        for cat, ratio in category_limits.items():

            recommended_limit = int(budget_limit * ratio)

            current_spending = expenses.get(cat, 0)

            print(cat, "→ Recommended:", recommended_limit)

            if current_spending > recommended_limit:

                excess = current_spending - recommended_limit

                print("⚠", cat, "is over the recommended limit by", excess)
                print("Suggestion: reduce spending in", cat, "by", excess)

            else:

                remaining = recommended_limit - current_spending

                print(cat, "is within limit. You can still spend", remaining)

        print("----------------------------\n")

        continue


    # -------- NORMAL TRANSACTION PROCESSING --------

    transactions = split_transactions(user_input)

    for transaction in transactions:

        user_vec = vectorizer.transform([transaction])
        prediction = model.predict(user_vec)[0]

        amount = extract_amount(transaction)

        print("\nTransaction:", transaction)
        print("Predicted Category:", prediction)
        print("Amount Detected:", amount)

        if prediction not in expenses:
            expenses[prediction] = 0

        expenses[prediction] += amount

        file_exists = os.path.isfile(history_file)

        with open(history_file, "a", newline="") as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["transaction", "amount", "category"])

            writer.writerow([transaction, amount, prediction])

    # Show summary
    print("\nCurrent Expense Summary:")

    for cat, val in expenses.items():
        print(cat, ":", val)

    # Dashboard visualization
    display_dashboard(expenses)

    # Budget warning
    if sum(expenses.values()) > budget_limit:
        print("\n⚠ Budget limit exceeded!\n")
    else:
        print("\nSpending is under control.\n")

    print("Transactions saved.\n")