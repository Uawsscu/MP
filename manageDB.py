#Python 2.7 /usr/bin/python2.7
import sqlite3
def create_Table(nameDB,exeText):
    con = sqlite3.connect(nameDB)
    c = con.cursor()
    #c.execute(exeText)
    c.execute('''CREATE TABLE obj_ALL(name varchar(50) primary key ,ID INTEGER)''')

#>>>>>>>>>>>>>> INSERT Obj Train <<<<<<<<<<<<<<<<

def insert_object_Train(name,lenID):
    with sqlite3.connect("Corpus_Main.db") as con:
        try :
            con.execute("insert into obj_ALL values (?, ?)", (name, lenID)) #nameObj,ID = {PK}
            print "ADD Object to Table obj_ALL : ", name," : ",lenID
        except :
            print "!!! DUPLICATE entry !!!"
            return "DUPLICATE"

#>>>>>>>>>>>>>> COUNT ROWs <<<<<<<<<<<<<<<<<<<<<<

def lenDB(nameDB,exeText):
    with sqlite3.connect(nameDB) as con:
        cur = con.cursor()
        cur.execute(exeText)
        rows = cur.fetchall()
        return len(rows)
        #print len(rows)

#>>>>>>>>>>>>> search obj TRAIN <<<<<<<<<<<<<<<<<<

def search_object_Train(name):
    with sqlite3.connect("Corpus_Main.db") as con:
        cur = con.cursor()
        #cur.execute('SELECT * FROM object_Train WHERE name=?',(name,))
        try :
            cur.execute("SELECT " + "ID" + " FROM obj_ALL where " + "name" + "=?", (name,))
            rows = cur.fetchone()
            for element in rows:
                return element
        except :
            return "None"
       # return cur.fetchone() # None OR (u'Ball', 2)


#>>>>>>>>>>>>>>>insert BUFF DETECT <<<<<<<<<<<<<
def insert_Buff_Detect(name):
    with sqlite3.connect("Corpus_Main.db") as con:
        try :
            con.execute("insert into Buff_Detect values (?, ?)", (name, 1)) #nameObj,ID = {PK}
        except :
            print "!!! DUPLICATE entry !!!"
            return "DUPLICATE"


#>>>>>>>>>>>>>>>remove BUFF DETECT <<<<<<<<<<<<<
def remove_Buff_Detect(id):
    with sqlite3.connect("Corpus_Main.db") as con:
        try :
           # con.execute('DELETE FROM Zoznam WHERE Name=?', (data,))
            con.execute("DELETE FROM Buff_Detect WHERE ID=?", (id,))
        except :
            print "!!! DUPLICATE entry !!!"
            return "DUPLICATE"

#>>>>>>>>>>>>> search Buff <<<<<<<<<<<<<<<<<<
def search_Buff_Detect(id):
    try :
        with sqlite3.connect("Corpus_Main.db") as con:
            cur = con.cursor()
            cur.execute("SELECT name FROM buff_detect WHERE ID=?", (id,))
            rows = cur.fetchone()
            for element in rows:
                return element
        # return cur.fetchone() # None OR (u'Ball', 2)
    except :
        return "None"

"""
if __name__ == '__main__':
    #create_1()
    lenObj = int(lenDB("Corpus_Main.db","SELECT * FROM object_Train"))
    #print search_object_Train("all")
    A= "ball"
    #print lenObj
    #print insertObj_NameNum(A,lenObj)   #/// DUPLICATE or NONE

print search_object_Train("Ball")
insert_Buff_Detect("ball")
remove_Buff_Detect(1)
"""
#insert_Buff_Detect("ball")
#print search_Buff_Detect(1)
#print search_object_Train("ball")
#lenObj = int(lenDB("Corpus_Main.db", "SELECT * FROM obj_ALL"))
#print lenObj
#insert_object_Train("teddy", int(lenObj+1))
#create_Table("Corpus_Main.db","'''CREATE TABLE obj_ALL(name varchar(50) ,ID INTEGER primary key)'''")


