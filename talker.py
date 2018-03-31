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


if __name__ == '__main__':
    Robot = raw_input("train or command : ")
    if (Robot == "train"):
        Action = str(raw_input("Enter you Action : "))
        while (True):
            step = input("Enter you Step : ")

            if step == 0:
                list = []
                for i in select_Buffer():
                    list.append(selectID_AcName(Action))
                    for x in i:
                        list.append(x)
                    with sqlite3.connect("Test_PJ2.db") as con:
                        cur = con.cursor()
                        cur.execute(
                            'insert into Action_Robot (ID,StepAction,M1,M2,M3,M4,M5,M6,M7,M8) values (?,?,?,?,?,?,?,?,?,?)',
                            (list))
                        print(list)
                        del list[:]
                del_buff()
                break

            talker(9)

            # with sqlite3.connect("Test_PJ2.db") as con:
            #  cur = con.cursor()
            #  cur.execute(
            #     'insert into Action_Robot (ID,StepAction,M1,M2,M3,M4,M5,M6,M7,M8) values (?,?,?,?,?,?,?,?,?,?)',
            #  (list))

            # list = []
            # del_buff()
            # break



            # if (check != {}):
            # del_buff()

    elif (Robot == "command"):
        Action = raw_input("Enter you Action : ")

        with sqlite3.connect("Test_PJ2.db") as con:
            cur = con.cursor()
            cur.execute(
                'Select Action_Robot.M1,Action_Robot.M2,Action_Robot.M3,Action_Robot.M4,Action_Robot.M5,Action_Robot.M6,Action_Robot.M7,Action_Robot.M8 from Action_Robot inner join ActionName on Action_Robot.ID = ActionName.ID where Name = ?',
                (Action,))
            row = cur.fetchall()

            for element in row:
                joint2 = str(element)
                # command1 = command1 + joint2
                command2 = joint2[1:]
                print(command2)
                talker1(command2)
                # joint = ""
                command1 = ""

                time.sleep(3)
                # break