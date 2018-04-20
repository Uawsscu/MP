

// modified by k. seepanomwan 2017-10-1
// read write to the dynamixel motor is ok via the DM74LS241N ic buffer


/*
   rosserial Publisher/Subscriber Example
   control the dynamixel motor callback

   on terminal#1: roscore
   on terminal#2: rosrun rosserial_python serial_node.py /dev/ttyUSB0

   // send message to the arduino via topic "setMotor"
   on terminal#3: rostopic pub setLED std_msgs/Empty --once ..or
                  rostopic pub setLED std_msgs/UInt8 0 --once ..or using the python script below

                  python talker.py 1, python talker.py 0
 
   // to see what's arduino published
   rostopic echo chatter ..or
   python listener.py
*/


#include "DynamixelMotor.h"

#include <ros.h>

#include <std_msgs/UInt8.h>
#include <std_msgs/String.h>

//std_msgs::UInt8 ui8_msg;
std_msgs::String str_msg;

ros::NodeHandle  nh;

SoftwareDynamixelInterface interface(3, 4, 2); // rx, tx, dir_control

DynamixelMotor motor1(interface, 1); // specify the dynamixel id
DynamixelMotor motor2(interface, 2);
DynamixelMotor motor3(interface, 3);
DynamixelMotor motor4(interface, 4);
DynamixelMotor motor5(interface, 5);
DynamixelMotor motor6(interface, 6);
DynamixelMotor motor7(interface, 7);
DynamixelMotor motor8(interface, 8);

//ros::Publisher pub1("ArduinoMsg", &ui8_msg);
ros::Publisher pub2("JointsValue", &str_msg);


void messageCb( const std_msgs::UInt8& msg) {

  //tone(8, 700);
  //delay(50);
  //noTone(8);

  // set the motor movement
  if (msg.data == 0) { // torque off
    motor1.torqueOn(0);
    motor2.torqueOn(0);
    motor3.torqueOn(0);
    motor4.torqueOn(0);
    motor5.torqueOn(0);
    motor6.torqueOn(0);
    motor7.torqueOn(0);
    motor8.torqueOn(0);
    
    //ui8_msg.data = 0;
    //pub1.publish( &ui8_msg );
  }

  if (msg.data == 1) { // torque on
    motor1.torqueOn(500);
    motor2.torqueOn(500);
    motor3.torqueOn(500);
    motor4.torqueOn(500);
    motor5.torqueOn(500);
    motor6.torqueOn(500);
    motor7.torqueOn(500);
    motor8.torqueOn(500);
    
    //ui8_msg.data = 1;
    //pub1.publish( &ui8_msg );
  }

  if (msg.data == 2) { // goto home position
    motor1.torqueOn(500);
    motor2.torqueOn(500);
    motor3.torqueOn(500);
    motor4.torqueOn(500);
    motor5.torqueOn(500);
    motor6.torqueOn(500);
    motor7.torqueOn(500);
    motor8.torqueOn(500);

    int data[8] = {181, 292, 180, 186, 205, 180, 189, 179};
    for(int i=0; i<8; i++){
      data[i] = map(data[i], 0, 360, 0, 1023);
    }

    motor1.goalPosition(data[0]);
    motor2.goalPosition(data[1]);
    motor3.goalPosition(data[2]);
    motor4.goalPosition(data[3]);
    motor5.goalPosition(data[4]);
    motor6.goalPosition(data[5]);
    motor7.goalPosition(data[6]);
    motor8.goalPosition(data[7]);
    
    //ui8_msg.data = 2;
    //pub1.publish( &ui8_msg );
  }

  if (msg.data == 3) { // goto Action1
    motor1.torqueOn(500);
    motor2.torqueOn(500);
    motor3.torqueOn(500);
    motor4.torqueOn(500);
    motor5.torqueOn(500);
    motor6.torqueOn(500);
    motor7.torqueOn(500);
    motor8.torqueOn(500);

    // step: 1
    int data1[8] = {186, 338, 176, 186, 217, 180, 261, 179};
    for(int i=0; i<8; i++){
      data1[i] = map(data1[i], 0, 360, 0, 1023);
    }

    motor1.goalPosition(data1[0]);
    motor2.goalPosition(data1[1]);
    motor3.goalPosition(data1[2]);
    motor4.goalPosition(data1[3]);
    motor5.goalPosition(data1[4]);
    motor6.goalPosition(data1[5]);
    motor7.goalPosition(data1[6]);
    motor8.goalPosition(data1[7]);
    delay(2000);

    // step: 2
    int data2[8] = {186, 337, 141, 165, 217, 179, 260, 179};
    for(int i=0; i<8; i++){
      data2[i] = map(data2[i], 0, 360, 0, 1023);
    }

    motor1.goalPosition(data2[0]);
    motor2.goalPosition(data2[1]);
    motor3.goalPosition(data2[2]);
    motor4.goalPosition(data2[3]);
    motor5.goalPosition(data2[4]);
    motor6.goalPosition(data2[5]);
    motor7.goalPosition(data2[6]);
    motor8.goalPosition(data2[7]);
    delay(2000);    

    // step: 3
    int data3[8] = {156, 275, 154, 165, 219, 179, 231, 207};
    for(int i=0; i<8; i++){
      data3[i] = map(data3[i], 0, 360, 0, 1023);
    }

    motor1.goalPosition(data3[0]);
    motor2.goalPosition(data3[1]);
    motor3.goalPosition(data3[2]);
    motor4.goalPosition(data3[3]);
    motor5.goalPosition(data3[4]);
    motor6.goalPosition(data3[5]);
    motor7.goalPosition(data3[6]);
    motor8.goalPosition(data3[7]);
    delay(1000);
    
    //ui8_msg.data = 3;
    //pub1.publish( &ui8_msg );
  }

  if (msg.data == 9) { // read current pose
    int m1 = map(motor1.currentPosition(),0,1023,0,360);
    int m2 = map(motor2.currentPosition(),0,1023,0,360);
    int m3 = map(motor3.currentPosition(),0,1023,0,360);
    int m4 = map(motor4.currentPosition(),0,1023,0,360);
    int m5 = map(motor5.currentPosition(),0,1023,0,360);
    int m6 = map(motor6.currentPosition(),0,1023,0,360);
    int m7 = map(motor7.currentPosition(),0,1023,0,360);
    int m8 = map(motor8.currentPosition(),0,1023,0,360);
    char cstr[50];
    
    String str1 = String( m1, DEC);
    String str2 = String( m2, DEC);
    String str3 = String( m3, DEC);
    String str4 = String( m4, DEC);
    String str5 = String( m5, DEC);
    String str6 = String( m6, DEC);
    String str7 = String( m7, DEC);
    String str8 = String( m8, DEC);
    
    String msg = str1+", "+str2+", "+str3+", "+str4+", "+str5+", "+str6+", "+str7+", "+str8;
    msg.toCharArray(cstr, 50);
    str_msg.data = cstr;
    pub2.publish( &str_msg );
  }

}

ros::Subscriber<std_msgs::UInt8> sub("setMotor", &messageCb );


void setup()
{
  pinMode(13, OUTPUT);
  digitalWrite(13, HIGH);

  delay(100);
  interface.begin(1000000); // default baud rate

  // motor configurations ----->>
  motor1.init();
  motor2.init();
  motor3.init();
  motor4.init();
  motor5.init();
  motor6.init();
  motor7.init();
  motor8.init();
  

  motor1.communicationSpeed(57600); // !!! IMPORTANCE !!!
  motor2.communicationSpeed(57600);
  motor3.communicationSpeed(57600);
  motor4.communicationSpeed(57600);
  motor5.communicationSpeed(57600);
  motor6.communicationSpeed(57600);
  motor7.communicationSpeed(57600);
  motor8.communicationSpeed(57600);

  interface.begin(57600); // !!! IMPORTANCE !!!

  delay(100);

  motor1.speed(40);
  motor2.speed(40);
  motor3.speed(40);
  motor4.speed(40);
  motor5.speed(40);
  motor6.speed(40);
  motor7.speed(40);
  motor8.speed(40);

  motor1.jointMode();
  motor2.jointMode();
  motor3.jointMode();
  motor4.jointMode();
  motor5.jointMode();
  motor6.jointMode();
  motor7.jointMode();
  motor8.jointMode();
  // <<----- motor configurations

  nh.initNode();
  
  nh.subscribe(sub);
  //nh.advertise(pub1);
  nh.advertise(pub2);

  digitalWrite(13, LOW);
}

void loop() {
  nh.spinOnce();
  delay(10);
}


