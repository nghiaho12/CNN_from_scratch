#include <iostream>
#include <fstream>

constexpr int WIDTH = 28;
constexpr int HEIGHT = 28;
constexpr int IMAGE_SIZE = WIDTH*HEIGHT;
constexpr int HEADER_SIZE = 12;

void display(char buf[IMAGE_SIZE]) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (static_cast<uint8_t>(buf[i*WIDTH + j]) >= 127) {
                std::cout << "#";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }  
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "./prog [images-ubyte] [image-num]\n";
        return 1;
    }

    std::ifstream is(argv[1], std::ios::binary);
    if (!is) {
        std::cerr << "can't open file: " << argv[1] << "\n";
        return 1;
    }

    int image_num = std::stoi(argv[2]);

    is.seekg(HEADER_SIZE + image_num*IMAGE_SIZE);

    char buf[IMAGE_SIZE];

    if (is.read(buf, IMAGE_SIZE)) {
        display(buf); 
    }

    return 0;
}
