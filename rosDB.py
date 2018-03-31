import sqlite3
#import numpy


def create_1():
    with sqlite3.connect("Test_PJ2.db") as con:
        AUTO_INCREMENT = 1
        sql_cmd = """
            create table ActionName(ID INTEGER, Name TEXT PRIMARY KEY)
        """
        con.execute(sql_cmd)


def create_2():
    with sqlite3.connect("Test_PJ2.db") as con:
        sql_cmd = """
            create table Buffer_Action(ID INTEGER PRIMARY KEY,M1 int,M2 int,M3 int,M4 int,M5 int,M6 int,M7 int,M8 int)

        """
        con.execute(sql_cmd)

def create_3():
    with sqlite3.connect("Test_PJ2.db") as con:
        sql_cmd = """
            create table Action_Robot(ID int,stepAction int,M1 int,M2 int,M3 int,M4 int,M5 int,M6 int,M7 int,M8 int)
        """
        con.execute(sql_cmd)


def drop_1():
    with sqlite3.connect("Test_PJ2.db") as con:
        sql_cmd = """
            drop table Buffer_Action;
        """
        con.execute(sql_cmd)

def insert_1():
    with sqlite3.connect("Test_PJ2.db") as con:
        sql_cmd = """
            insert into ActionName(Name,ID) values ('grab sports ball',1),('touch cup',2),('move the teddy bear to the left',3),('move the bottle to the right',4)
        """
        con.execute(sql_cmd)

def insert_2():
    with sqlite3.connect("Test_PJ2.db") as con:
        sql_cmd = """
            insert into Action_Robot(ID,StepAction,M1,M2,M3,M4,M5,M6,M7,M8) values (2,1,186, 338, 176, 186, 217, 180, 261, 179),(2,2,186, 337, 141, 165, 217, 179, 260, 179),(2,3,156, 275, 154, 165, 219, 179, 231, 207)
        """
        con.execute(sql_cmd)



if __name__ == '__main__':
    #drop_1()
    create_2()
    #create_3()
    #insert_2()
