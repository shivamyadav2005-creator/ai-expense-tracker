import streamlit as st
import pandas as pd
import re
import os
import csv
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("Data/transactions_1000_clean.csv")

X = data["text"]
y = data["category"]

vectorizer = CountVectorizer(ngram_range=(1,2))
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# extract amount
def extract_amount(text):
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return 0

# split transactions
def split_transactions(text):
    matches = re.findall(r'[a-zA-Z\s]+?\s\d+', text)

    transactions = []

    for m in matches:
        transactions.append(m.strip())

    if len(transactions) == 0:
        transactions.append(text)

    return transactions

# page title
st.title("💰 AI Expense Tracker")

# budget
budget_limit = 5000

# recommended category limits
category_limits = {
    "food": 0.30,
    "transport": 0.20,
    "entertainment": 0.20,
    "shopping": 0.30
}

# history file
history_file = "history.csv"

# session storage
if "expenses" not in st.session_state:
    st.session_state.expenses = {}

user_input = st.text_input("Enter transaction")
transaction_date = st.date_input("Select transaction date")

# ADD TRANSACTION
if st.button("Add Transaction"):

    transactions = split_transactions(user_input)

    for transaction in transactions:

        user_vec = vectorizer.transform([transaction])
        prediction = model.predict(user_vec)[0]

        amount = extract_amount(transaction)

        if prediction not in st.session_state.expenses:
            st.session_state.expenses[prediction] = 0

        st.session_state.expenses[prediction] += amount

        st.success(f"{transaction} → {prediction} : {amount}")

        # SAVE HISTORY
        file_exists = os.path.isfile(history_file)

        with open(history_file, "a", newline="") as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["transaction","amount","category","date"])

            writer.writerow([transaction, amount, prediction, transaction_date])

# Show summary
if st.session_state.expenses:

    st.subheader("Expense Summary")

    df = pd.DataFrame(
        list(st.session_state.expenses.items()),
        columns=["Category", "Amount"]
    )

    st.table(df)

    total = df["Amount"].sum()

    st.write("### Total Spending:", total)

    remaining_budget = budget_limit - total
    highest_category = df.loc[df["Amount"].idxmax()]

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Spent", total)
    col2.metric("Remaining Budget", remaining_budget)
    col3.metric("Top Category", highest_category["Category"])

    st.write(
        f"### Highest Spending Category: **{highest_category['Category']}** "
        f"({highest_category['Amount']})"
    )

    if total > budget_limit:
        st.error("⚠ Budget limit exceeded!")
    else:
        st.success("Spending under control")

    # -------- PERCENTAGE BREAKDOWN --------

    st.subheader("Category Breakdown")

    df["Percentage"] = (df["Amount"] / total) * 100

    st.table(df)

    # -------- EXPENSE DISTRIBUTION CHART --------

    st.subheader("Expense Distribution")

    st.bar_chart(df.set_index("Category")["Amount"])

    # -------- PIE CHART --------

    st.subheader("🥧 Category Spending Share")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.pie(
        df["Amount"],
        labels=df["Category"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.axis("equal")

    st.pyplot(fig)

    # -------- DOWNLOAD CSV REPORT --------

    st.subheader("⬇ Download Spending Report")

    csv_report = df.to_csv(index=False)

    st.download_button(
        label="Download CSV Report",
        data=csv_report,
        file_name="expense_report.csv",
        mime="text/csv"
    )

    # -------- PDF REPORT --------

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    import io

    pdf_buffer = io.BytesIO()

    c = canvas.Canvas(pdf_buffer, pagesize=letter)

    c.setFont("Helvetica", 12)

    c.drawString(200, 750, "AI Expense Tracker Report")

    y = 700

    for index, row in df.iterrows():
        line = f"{row['Category']} : {row['Amount']}"
        c.drawString(100, y, line)
        y -= 20

    c.drawString(100, y-20, f"Total Spending: {total}")

    c.save()

    pdf_buffer.seek(0)

    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="expense_report.pdf",
        mime="application/pdf"
    )

    # -------- BUDGET SUGGESTIONS --------

    st.subheader("Budget Recommendations")

    for cat, ratio in category_limits.items():

        recommended_limit = int(budget_limit * ratio)

        current_spending = st.session_state.expenses.get(cat, 0)

        st.write(f"**{cat.upper()} → Recommended Limit: {recommended_limit}**")

        if current_spending > recommended_limit:

            excess = current_spending - recommended_limit

            st.error(
                f"{cat.capitalize()} spending is over the limit by {excess}. "
                f"Reduce spending in {cat} by {excess}."
            )

        else:

            remaining = recommended_limit - current_spending

            st.success(
                f"{cat.capitalize()} is within limit. "
                f"You can still spend {remaining}."
            )

    # -------- OVESPENDING PREDICTION --------

    st.subheader("📉 Overspending Prediction")

    if os.path.exists(history_file):

        history_df = pd.read_csv(history_file, on_bad_lines="skip")

        history_df["date"] = pd.to_datetime(history_df["date"], format="mixed", errors="coerce")

        days_recorded = history_df["date"].dt.date.nunique()

        if days_recorded == 0:
            days_recorded = 1

        total_spent = history_df["amount"].sum()

        daily_rate = total_spent / days_recorded

        if daily_rate > 0:

            remaining_budget = budget_limit - total_spent

            predicted_days = int(remaining_budget / daily_rate)

            if remaining_budget < 0:
                st.error("Budget already exceeded.")
            else:
                st.warning(
                    f"At current spending rate you may exceed your budget in **{predicted_days} days**."
                )

    # -------- SMART SPENDING SUGGESTIONS --------

    st.subheader("🧠 Smart Spending Suggestions")

    if total > budget_limit:

        excess = total - budget_limit

        st.write(f"You exceeded budget by **{excess}**")

        df_sorted = df.sort_values(by="Amount", ascending=False)

        top_categories = df_sorted.head(3)

        for index, row in top_categories.iterrows():

            reduction = int(excess * (row["Amount"] / total))

            daily_cut = int(reduction / 30)

            st.error(
                f"Reduce **{row['Category']}** by about ₹{daily_cut}/day "
                f"(₹{reduction} monthly)"
            )

    else:

        st.success("Your spending pattern is healthy.")

    # -------- AI SPENDING INSIGHTS --------

    st.subheader("🤖 AI Spending Insights")

    top_category = df.loc[df["Amount"].idxmax()]
    top_percent = (top_category["Amount"] / total) * 100

    if top_percent > 40:
        st.warning(
            f"You are spending **{top_percent:.1f}%** of your money on "
            f"**{top_category['Category']}**."
        )

    for index, row in df.iterrows():

        percent = (row["Amount"] / total) * 100

        if percent > 35:
            st.error(
                f"High spending detected in **{row['Category']}** "
                f"({percent:.1f}% of total spending)."
            )

        elif percent < 10:
            st.success(
                f"Good control on **{row['Category']}** spending "
                f"({percent:.1f}% of total)."
            )

    if total > budget_limit:

        excess = total - budget_limit
        daily_reduction = int(excess / 30)

        st.error(
            f"Reduce spending by approximately **₹{daily_reduction}/day** next month."
        )

    else:

        remaining = budget_limit - total
        daily_allowance = int(remaining / 30)

        st.info(
            f"You can safely spend about **₹{daily_allowance}/day** for the rest of the month."
        )

    # -------- SPENDING TREND --------

    st.subheader("📊 Spending Trend Over Time")

    if os.path.exists(history_file):

        history_df = pd.read_csv(history_file, on_bad_lines="skip")

        history_df["date"] = pd.to_datetime(history_df["date"], format="mixed", errors="coerce")
        trend = history_df.groupby(history_df["date"].dt.date)["amount"].sum()

        st.write("### Daily Spending Trend")
        st.line_chart(trend)

        history_df = history_df.sort_values("date")
        history_df["cumulative_spending"] = history_df["amount"].cumsum()

        st.write("### Cumulative Spending Growth")
        st.line_chart(history_df.set_index("date")["cumulative_spending"])

    else:

        st.info("No history available yet.")

# -------- HISTORY VIEWER --------

if st.button("Show Transaction History"):

    if os.path.exists(history_file):

        history_df = pd.read_csv(history_file, on_bad_lines="skip")

        st.subheader("Transaction History")

        st.dataframe(history_df)

    else:

        st.warning("No transaction history found.")

