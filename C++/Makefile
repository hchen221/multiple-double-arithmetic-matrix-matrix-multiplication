all: matmatTC

matmatTC:
	g++ -c floatybits.cpp TCflat.cpp
	g++ -c floatybits.cpp TCtest.cpp
	g++ -c TCflat.cpp TCtest.cpp
	g++ floatybits.o TCflat.o TCtest.o