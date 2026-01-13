import sqlite3
from datetime import datetime

#-----------------------------------------------------------------------
# Class that provides initialization and CRUD operations for a time-off
# database using SQLite
#-----------------------------------------------------------------------
class TimeOffDatastore:
    #Initialize the database connection, create tables and seed data
    def __init__(self, db_path=":memory:"):
        print("Initializing TimeOffDatastore")
        self.conn = sqlite3.connect(db_path)
        print("Creating tables and seeding data")
        self.create_tables()
        self.seed_data()

    # Create tables for employee and timeoff history
    def create_tables(self):
        cursor = self.conn.cursor()

        # employee table tracks time off balance also
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employee (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                allowed_days INTEGER NOT NULL,
                consumed_days INTEGER NOT NULL DEFAULT 0
            )
        ''')

        # timeoff_history table tracks time off requests
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeoff_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                start_day TEXT NOT NULL,
                total_days INTEGER NOT NULL,
                FOREIGN KEY(employee_id) REFERENCES employee(id)
            )
        ''')
        self.conn.commit()

    # Seed the database with initial data
    def seed_data(self):
        cursor = self.conn.cursor()
        # Insert sample employees if not already present
        employees = [
            ("Alice", 20, 5),
            ("Bob", 15, 3),
            ("Charlie", 25, 10)
        ]
        for name, allowed, consumed in employees:
            cursor.execute('''
                INSERT OR IGNORE INTO employee (name, allowed_days, consumed_days)
                VALUES (?, ?, ?)
            ''', (name, allowed, consumed))
        self.conn.commit()

    # Get timeoff balance for a specific employee
    def get_timeoff_balance(self, employee_name):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT allowed_days, consumed_days FROM employee WHERE name = ?
        ''', (employee_name,))
        row = cursor.fetchone()
        print("Row fetched: ", row)
        if row:
            allowed, consumed = row
            return str(allowed - consumed)
        else:
            return "Employee not found"

    # Add a timeoff request for an employee
    # This function checks if the employee has enough timeoff balance
    # and updates the timeoff history and employee's consumed days
    def add_timeoff_request(self, employee_name, start_day, total_days):
        cursor = self.conn.cursor()

        # Find employee ID and current consumed_days
        cursor.execute('''
            SELECT id, allowed_days, consumed_days FROM employee WHERE name = ?
        ''', (employee_name,))
        row = cursor.fetchone()
        print("Row fetched: ", row)
        if not row:
            return "Error: Employee not found"
        emp_id, allowed, consumed = row
        if consumed + total_days > allowed:
            return "Error: Not enough timeoff balance"

        # Insert into timeoff_history
        cursor.execute('''
            INSERT INTO timeoff_history (employee_id, start_day, total_days)
            VALUES (?, ?, ?)
        ''', (emp_id, start_day, total_days))

        # Update consumed_days
        cursor.execute('''
            UPDATE employee SET consumed_days = consumed_days + ?
            WHERE id = ?
        ''', (total_days, emp_id))
        self.conn.commit()
        return "Successfully added timeoff request"

# Example usage:
if __name__ == "__main__":
    ds = TimeOffDatastore()
    print("Alice's balance:", ds.get_timeoff_balance("Alice"))
    ds.add_timeoff_request("Alice", "2024-06-10", 2)
    print("Alice's balance after request:", 
            ds.get_timeoff_balance("Alice"))