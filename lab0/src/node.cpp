#include "ros/ros.h"
#include "std_msgs/String.h"

class TestNode {
private:
    ros::NodeHandle nh;
    ros::Publisher pub;
    ros::Subscriber sub;

public:
    TestNode() {
        // 初始化 Publisher：發送到 /output_topic
        pub = nh.advertise<std_msgs::String>("/output_topic", 10);
        
        // 初始化 Subscriber：接收 /input_topic
        sub = nh.subscribe("/input_topic", 10, &TestNode::callback, this);
    }

    void callback(const std_msgs::String::ConstPtr& msg) {
        ROS_INFO("I heard: [%s]", msg->data.c_str());
        
        // 收到訊息後，加工再發送出去
        std_msgs::String out_msg;
        out_msg.data = "Processed: " + msg->data;
        pub.publish(out_msg);
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "test_node");
    TestNode node_obj;
    ros::spin(); // 持續監聽
    return 0;
}