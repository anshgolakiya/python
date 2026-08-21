import mysql.connector
con = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="ansh55"
)
print(con)
cursor = con.cursor()
cursor.execute("DROP TABLE IF EXISTS student")
cursor.execute("""CREATE TABLE student(
    id INT PRIMARY KEY ,
    name VARCHAR(20),
    marks INT
)""")
print("Table created successfully.\n")
con.commit()
sql = "INSERT INTO student(id, name, marks) VALUES(%s, %s, %s)"
data = (1, "Ansh", 85)
cursor.execute(sql, data)
print("record inserted successfully.\n")
con.commit()

sql= "INSERT INTO student(id, name, marks) VALUES(%s, %s, %s)"
data = [
    (2, "Raj", 80) ,
    (3, "Priya", 90) ,
    (4, "Rahul", 75)
]
cursor.executemany(sql,data)
print("more record inserted successfully.\n")
con.commit()

cursor.execute("SELECT * FROM student")
print("\nStudent Records :")
for row in cursor.fetchall() :
    print(row)

cursor.execute("SELECT name, marks FROM student")
print("\nName and marks of students :")
rows = cursor.fetchall()
for row in rows :
    print(row)

cursor.execute("UPDATE student SET marks = %s WHERE id = %s",(90,1))
con.commit()
print("Records update successfully\n")

cursor.execute("SELECT * FROM student WHERE id = 1")
print(cursor.fetchone())

sql = "DELETE FROM student WHERE id = %s"
data = (4, )
cursor.execute(sql,data)
con.commit()
print("Record delete successfully.\n")

cursor.execute("SELECT * FROM student")
print("\nStudent Records :")
for row in cursor.fetchall() :
    print(row)

cursor.close()
con.close()