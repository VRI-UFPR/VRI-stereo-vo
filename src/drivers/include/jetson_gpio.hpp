#ifndef __JETSON_GPIO__
#define __JETSON_GPIO__

#include <string>
#include <fstream>
#include <iostream>

class JetsonGPIO
{
public:
    enum Direction
    {
        INPUT,
        OUTPUT
    };

    enum Value
    {
        LOW,
        HIGH
    };
    
    JetsonGPIO(){}
    JetsonGPIO(int pin, JetsonGPIO::Direction direction, JetsonGPIO::Value initial_value);
    ~JetsonGPIO();
    JetsonGPIO& operator=(const JetsonGPIO& t);

    void setValue(JetsonGPIO::Value value);

    int pin;
private:
    const std::string gpio_path = "/sys/class/gpio";
    
    JetsonGPIO::Direction direction;
    JetsonGPIO::Value curr_value;
    std::string pin_path;
};

JetsonGPIO::JetsonGPIO(int pin, JetsonGPIO::Direction direction, JetsonGPIO::Value initial_value)
{
    // set properties
    this->pin = pin;
    this->direction = direction;
    this->curr_value = initial_value;
    this->pin_path = this->gpio_path + "/gpio" + std::to_string(pin);

    // export pin
    std::string command = "echo " + std::to_string(this->pin) + " > " + this->gpio_path + "/export";
    std::cout << command << std::endl;
    system(command.c_str());
    // std::ofstream export_file(this->gpio_path + "/export");
    // export_file << this->pin;
    // export_file.close();

    // set direction
    // std::ofstream direction_file(this->pin_path + "/direction");
    std::string direction_str = (direction == JetsonGPIO::Direction::INPUT ? "in" : "out");
    command = "echo " + direction_str + " > " + this->pin_path + "/direction";
    std::cout << command << std::endl;
    system(command.c_str());

    // set initial value
    this->setValue(initial_value);
}

JetsonGPIO::~JetsonGPIO()
{
    // unexport pin
    std::string command = "echo " + std::to_string(this->pin) + " > " + this->gpio_path + "/unexport";
    std::cout << command << std::endl;
    system(command.c_str());
    // std::ofstream unexport_file(this->gpio_path + "/unexport");
    // unexport_file << this->pin;
    // unexport_file.close();
}

JetsonGPIO& JetsonGPIO::operator=(const JetsonGPIO& t)
{
    this->curr_value = t.curr_value;
    this->direction = t.direction;
    this->pin_path = t.pin_path;
    this->pin = t.pin;

    return *this;
}

void JetsonGPIO::setValue(JetsonGPIO::Value value)
{
    std::string command = "echo " + std::to_string(value) + " > " + this->pin_path + "/value";
    std::cout << command << std::endl;
    system(command.c_str());

    // std::ofstream value_file(this->pin_path + "/value");
    // value_file << value;
    // value_file.close();
}

#endif // __JETSON_GPIO__