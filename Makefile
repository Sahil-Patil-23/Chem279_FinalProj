# Compile Code
vibes:
	cd src && g++ -std=c++11 -I /usr/local/include/eigen-3.4.0 vibrational_analysis.cpp -o vibrational_analysis

# Run executable
run: 
	cd src && ./vibrational_analysis

# Compile AND Run
all:
	make vibes && make run