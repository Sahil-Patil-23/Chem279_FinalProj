#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
using namespace std;

struct Atom{
    double mass;
    vector<double> coordinates;
};

// Function that calculates distance between 
double Calculate_Distance(const vector<double> &atom1, const vector<double> &atom2){
    return sqrt( pow(atom1[0] - atom2[0], 2) +
                 pow(atom1[1] - atom2[1], 2) +
                 pow(atom1[2] - atom2[2], 2));
}

// Function that calculates the Lennard Jones potential between 2 atoms 
double Calculate_LJ(double distance, double Sigma, double Epsilon){
    double sig_dist = Sigma/distance;
    double term6 = pow(sig_dist, 6);
    double term12 = pow(sig_dist, 12);

    return (Epsilon - (term12 - (2 * term6)));
}

// Function that calculates the total energy of a cluster of atoms
double Calculate_Total_Energy(const vector<Atom> &atoms, double Sigma, double Epsilon){
    double total_energy = 0.0;

    for(size_t i = 0; i < atoms.size(); i++){
        for(size_t j = i + 1; j < atoms.size(); j++){
            double distance = Calculate_Distance(atoms[i].coordinates , atoms[j].coordinates);
            double LJ = Calculate_LJ(distance, Sigma, Epsilon);
            total_energy += LJ;
        }
    }

    return total_energy;
}

// Function that computes the Hessian Matrix
Eigen::MatrixXd Compute_Hessian_Matrix(vector<Atom> &atoms, double delta, double Sigma, double Epsilon){
    int num_atoms = atoms.size();
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(3 * num_atoms, 3 * num_atoms);

    for (int i = 0; i < num_atoms; ++i) {
        for (int j = 0; j < 3; ++j) {
            // Perturb the coordinate in positive and negative directions
            atoms[i].coordinates[j] += delta;
            double energy_plus = Calculate_Total_Energy(atoms, Sigma, Epsilon);

            atoms[i].coordinates[j] -= 2 * delta;
            double energy_minus = Calculate_Total_Energy(atoms, Sigma, Epsilon);

            // Restore the original coordinate
            atoms[i].coordinates[j] += delta;

            // Compute second derivative (Hessian element)
            double second_derivative = (energy_plus - 2 * Calculate_Total_Energy(atoms, Sigma, Epsilon) + energy_minus) / (delta * delta);
            hessian(3 * i + j, 3 * i + j) = second_derivative;
        }
    }
    return hessian;
}

int main(){
    return 0;
}