import sqlite3

conn = sqlite3.connect('gold_high_iq.db')
cursor = conn.cursor()

cursor.execute('PRAGMA table_info(gold_high_iq)')

print("\nDatabase Schema:")
print("-" * 80)
for row in cursor.fetchall():
    print(f"{row[1]:25s} {row[2]:15s}")

conn.close()
