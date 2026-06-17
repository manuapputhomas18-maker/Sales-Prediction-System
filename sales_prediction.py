import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# MYSQL CONNECTION
# =========================

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="sales_db"
)

cursor = conn.cursor()

# =========================
# FETCH DATA
# =========================

cursor.execute("""
SELECT pro_name,
       advertisement,
       discount,
       customers,
       sales
FROM sales
""")

data = cursor.fetchall()

# =========================
# DATAFRAME
# =========================

df = pd.DataFrame(
    data,
    columns=[
        "pro_name",
        "advertisement",
        "discount",
        "customers",
        "sales"
    ]
)

print("\nPRODUCT SALES DATA\n")
print(df)

# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(8,5))

plt.bar(
    df["pro_name"],
    df["sales"]
)

plt.xticks(rotation=45)

plt.title("Product Sales")

plt.xlabel("Pro Name")

plt.ylabel("Sales")

plt.show()

# =========================
# FEATURES
# =========================

X = df[[
    "advertisement",
    "discount",
    "customers"
]]

y = df["sales"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# TRAIN MODEL
# =========================

model = LinearRegression()
model.fit(X_train,y_train)

print("\nModel Trained Successfully")

# PREDICTION
# =========================

predictions = model.predict(X_test)
result = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": predictions
})

print("\nPrediction Result\n")
print(result)

# =========================
# ACTUAL VS PREDICTED GRAPH
# =========================

plt.figure(figsize=(6,4))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# =========================
# ACCURACY
# =========================

score = r2_score(y_test,predictions)
print("\nR2 Score :", round(score,2))

# =========================
# USER INPUT
# =========================

print("\nNEW PRODUCT SALES PREDICTION")

product_name = input( "Enter Product Name : ")

advertisement = float(input("Advertisement Budget : "))

discount = float(input("Discount Percentage : "))

customers = int(input("Expected Customers : "))

new_data = [[
    advertisement,
    discount,
    customers
]]

predicted_sales = model.predict(new_data)

print("\nProduct :", product_name)

print("Predicted Sales : ₹",round(predicted_sales[0],2))

# =========================
# CLOSE CONNECTION
# =========================

cursor.close()
conn.close()