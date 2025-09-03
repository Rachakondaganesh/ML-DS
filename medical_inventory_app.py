
import pymysql
import datetime

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "ganesh",
    "db": "medical_db"
}

# Create table if not exists
def initialize_database():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medicines (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            batch_no TEXT NOT NULL,
            expiry_date DATE NOT NULL,
            quantity INT NOT NULL,
            price FLOAT NOT NULL,
            supplier TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Add new medicine
def add_medicine(name, batch_no, expiry_date, quantity, price, supplier):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO medicines (name, batch_no, expiry_date, quantity, price, supplier)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (name, batch_no, expiry_date, quantity, price, supplier))
    conn.commit()
    conn.close()
    print("✅ Medicine added successfully.")

# View all medicines
def view_medicines():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM medicines")
    rows = cursor.fetchall()
    print("\n📋 All Medicines:")
    for row in rows:
        print(row)
    conn.close()

# Search medicine by keyword
def search_medicine(keyword):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM medicines
        WHERE name LIKE %s OR batch_no LIKE %s
    ''', ('%' + keyword + '%', '%' + keyword + '%'))
    rows = cursor.fetchall()
    print(f"\n🔍 Search Results for '{keyword}':")
    for row in rows:
        print(row)
    conn.close()

# Update stock quantity
def update_stock(medicine_id, quantity_change):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE medicines
        SET quantity = quantity + %s
        WHERE id = %s
    ''', (quantity_change, medicine_id))
    conn.commit()
    conn.close()
    print("📦 Stock updated successfully.")

# Check for expired medicines
def check_expiry():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    today = datetime.date.today()
    cursor.execute('''
        SELECT * FROM medicines
        WHERE expiry_date <= %s
    ''', (today,))
    rows = cursor.fetchall()
    print("\n⚠️ Expired Medicines:")
    for row in rows:
        print(row)
    conn.close()

# Check for low stock
def check_low_stock(threshold=10):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM medicines
        WHERE quantity <= %s
    ''', (threshold,))
    rows = cursor.fetchall()
    print(f"\n📉 Medicines with Stock ≤ {threshold}:")
    for row in rows:
        print(row)
    conn.close()

# Initialize database
initialize_database()

# Main menu loop
while True:
    print("\n---- MEDICAL INVENTORY MENU ----")
    print("1. Add Medicine")
    print("2. View Medicines")
    print("3. Search Medicine")
    print("4. Update Stock")
    print("5. Check Expiry")
    print("6. Check Low Stock")
    print("7. Exit")

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        name = input("Name: ")
        batch = input("Batch No: ")
        expiry = input("Expiry Date (YYYY-MM-DD): ")
        quantity = int(input("Quantity: "))
        price = float(input("Price: "))
        supplier = input("Supplier: ")
        add_medicine(name, batch, expiry, quantity, price, supplier)

    elif choice == "2":
        view_medicines()

    elif choice == "3":
        keyword = input("Enter keyword to search: ")
        search_medicine(keyword)

    elif choice == "4":
        medicine_id = int(input("Enter Medicine ID: "))
        qty_change = int(input("Enter quantity to add/remove (+/-): "))
        update_stock(medicine_id, qty_change)

    elif choice == "5":
        check_expiry()

    elif choice == "6":
        threshold = input("Enter stock threshold (default 10): ")
        threshold = int(threshold) if threshold else 10
        check_low_stock(threshold)

    elif choice == "7":
        print("👋 Thank you for using the Medical Inventory System.")
        break

    else:
        print("❌ Invalid choice. Please try again.")





