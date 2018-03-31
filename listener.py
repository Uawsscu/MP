#!/usr/bin/env phthon
import numpy as np
import sqlite3
import rospy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import UInt8
from std_msgs.msg import UInt16
#from std_msgs.msg import String
#from sensor_msgs.msg import JointState

step = 0
D1 = []
def type_data(data):
   A = data.data
#   B = A.split(',')
   print(A)
   print(type(A))

def callback_List(data):
   with sqlite3.connect("Test_PJ2.db") as con:
      cur = con.cursor()
   A = data.data
   B = A.split(',')
   if len(D1) == 0:
      cur.execute('select * from Buffer_Action')
      rows = cur.fetchall()
      D1.append(len(rows) + 1)
   for i in B:
      C = int(i)
      D1.append(C)
   print(D1)
   if len(D1) == 9:
      with sqlite3.connect("Test_PJ2.db") as con:
         cur = con.cursor()
         cur.execute('insert into Buffer_Action (ID,M1,M2,M3,M4,M5,M6,M7,M8) values (?,?,?,?,?,?,?,?,?)' ,(D1))

         del D1[:]


      #  print(D)
      #  with sqlite3.connect("Test_PJ2.db") as con:
      #     cur = con.cursor()
      #     cur.execute('insert into Buffer_Action (ID,M1,M2,M3,M4,M5,M6,M7,M8) values (?,?,?,?,?,?,?,?,?)', (D))
      #  D = []

         #del D[:]


def listener():
   rospy.init_node('Listener', anonymous=True)
   rospy.Subscriber("ArduinoMsg", UInt16, type_data)
   rospy.Subscriber("JointsValue", String, callback_List)
   #rospy.Subscriber("ArduinoMsg", UInt16, callback)
   rospy.spin()

if __name__ == '__main__':
   listener()
   #callback_List("11,22,33,44,55,66,77,88")