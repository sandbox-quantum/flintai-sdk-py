"""
Expense Tracker Agent using Google ADK
Helps users maintain financial records through natural conversation
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

from flintai.plugins.adk import ADKGuardrailsPlugin
from google.adk.agents import LlmAgent

# Database setup
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expenses.db")


def init_db():
    """Initialize SQLite database with expenses table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def add_expense(date: str, amount: float, category: str, description: str = "") -> str:
    """
    Add an expense to the database

    Args:
        date: Date in YYYY-MM-DD format
        amount: Amount spent (positive number)
        category: Category of expense (e.g., food, transport, entertainment)
        description: Optional description

    Returns:
        Confirmation message
    """
    try:
        # Validate date format
        datetime.strptime(date, "%Y-%m-%d")

        # Validate amount
        if amount <= 0:
            return "Error: Amount must be positive"

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO expenses (date, amount, category, description) VALUES (?, ?, ?, ?)",
            (date, amount, category, description),
        )
        conn.commit()
        conn.close()

        return f"✓ Added ${amount:.2f} expense for {category} on {date}"
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD"
    except Exception as e:
        return f"Error adding expense: {str(e)}"


def get_monthly_spending(year: int, month: int) -> str:
    """
    Get total spending for a specific month

    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)

    Returns:
        Monthly spending summary
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get month pattern for SQLite
        month_str = f"{year}-{month:02d}%"

        # Total for month
        cursor.execute(
            "SELECT SUM(amount) FROM expenses WHERE date LIKE ?", (month_str,)
        )
        total = cursor.fetchone()[0] or 0

        # Breakdown by category
        cursor.execute(
            "SELECT category, SUM(amount), COUNT(*) FROM expenses"
            " WHERE date LIKE ?"
            " GROUP BY category ORDER BY SUM(amount) DESC",
            (month_str,),
        )
        breakdown = cursor.fetchall()

        conn.close()

        result = f"📊 Spending for {year}-{month:02d}\n"
        result += f"Total: ${total:.2f}\n\n"

        if breakdown:
            result += "By Category:\n"
            for cat, amt, count in breakdown:
                result += f"  • {cat}: ${amt:.2f} ({count} transactions)\n"
        else:
            result += "No expenses recorded for this month"

        return result
    except Exception as e:
        return f"Error getting monthly spending: {str(e)}"


def get_current_date() -> str:
    """
    Get the current date.

    Returns:
        The current date as a formatted string
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d (%A)")


def search_expenses(
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    Search expenses with optional filters

    Args:
        category: Filter by category
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Maximum number of results

    Returns:
        List of matching expenses
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = "SELECT date, amount, category, description FROM expenses WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No expenses found matching your criteria"

        result = f"📝 Found {len(rows)} expense(s):\n\n"
        for date, amount, cat, desc in rows:
            result += f"• {date} - ${amount:.2f} ({cat})"
            if desc:
                result += f": {desc}"
            result += "\n"

        return result
    except Exception as e:
        return f"Error searching expenses: {str(e)}"


# Create the expense tracker agent using ADK
def create_expense_agent():
    """Create and configure the expense tracking agent"""
    system_instruction = """You are a helpful expense tracking assistant.

When a user mentions an expense:
1. Extract the date (default to today if not specified), amount, and description
2. Ask the user to confirm the category, amount, and date before saving
3. Only call add_expense after explicit user confirmation

When asked about spending:
- Use get_monthly_spending for monthly summaries
- Use search_expenses for filtered queries

Use get_current_date to look up today's date when needed."""

    plugin = ADKGuardrailsPlugin()

    agent = LlmAgent(
        name="expense_tracker",
        model="gemini-3-flash-preview",
        instruction=system_instruction,
        description="An AI assistant that helps users track and analyze their expenses",
        tools=[get_current_date, add_expense, get_monthly_spending, search_expenses],
        generate_content_config=plugin.content_config,
        before_model_callback=plugin.before_model_callback,
        on_model_error_callback=plugin.on_model_error,
    )

    return agent


init_db()
root_agent = create_expense_agent()
