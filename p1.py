###########################################################################################################
import sys
import rospy
import sqlite3
import time
from std_msgs.msg import UInt8
from std_msgs.msg import String
from std_msgs.msg import UInt16


def callback_Talk(msg):  # insert ActionName
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute('SELECT * FROM ActionName')
            rows = cur.fetchall()
            lenR = len(rows)
            cur.execute('insert into ActionName (Name,ID) values (?,?)', (msg, lenR + 1,))
        except:
            return "Name Error"


def selectID_AcName(Action):
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute("Select ID from ActionName where Name = ?", (Action,))
            row = cur.fetchone()
            for element in row:
                id = element
                return id
        except:
            return "Error"


def select_Buffer():
    list = []
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute("Select ID,M1,M2,M3,M4,M5,M6,M7,M8 from Buffer_Action")
            row = cur.fetchall()
            for element in row:
                motor = element
                list.append(motor)
            return list
        except:
            return "Null"


def del_buff():
    with sqlite3.connect("Test_PJ2.db") as con:
        sql_cmd = """
        delete from Buffer_Action

        """
        con.execute(sql_cmd)


def talker(msg):
    pub1 = rospy.Publisher('setMotor', UInt16, queue_size=10)

    rospy.init_node('Talker', anonymous=True)

    pub1.publish(int(msg))


def talker1(msg1):
    pub2 = rospy.Publisher('joints', String, queue_size=10)

    rospy.init_node('Talker', anonymous=True)

    pub2.publish(String(msg1))


def talker2(msg2):
    pub3 = rospy.Publisher('walker', String, queue_size=10)

    rospy.init_node('Talker', anonymous=True)

    pub3.publish(String(msg2))


################################################## Keep #####################################################

talker2("580")