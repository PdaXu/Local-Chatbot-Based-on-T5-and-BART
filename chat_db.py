import sqlite3
from datetime import datetime

DB_PATH = r'D:\study\MGW\ChatAPP\chat_history.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            user_input TEXT NOT NULL,
            model_reply TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_record(model, user_input, model_reply):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO history (model, user_input, model_reply, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (model, user_input, model_reply, timestamp))
    conn.commit()
    conn.close()

def get_all_records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows
