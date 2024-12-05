#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
using namespace std;

struct Atom {
    double mass;                 // Atomic mass
    vector<double> coordinates;  // x, y, z coordinates
};

// Function to calculate the distance between two atoms
double Calculate_Distance(const vector<double> &atom1, const vector<double> &atom2) {
    return sqrt(pow(atom1[0] - atom2[0], 2) +
                pow(atom1[1] - atom2[1], 2) +
                pow(atom1[2] - atom2[2], 2));
}

// Function to calculate the Lennard-Jones potential between two atoms
double Calculate_LJ(double distance, double Sigma, double Epsilon) {
    double sig_dist = Sigma / distance;
    double term6 = pow(sig_dist, 6);
    double term12 = pow(sig_dist, 12);
    return 4 * Epsilon * (term12 - term6);
}

// Function to calculate the total energy of a cluster of atoms
double Calculate_Total_Energy(const vector<Atom> &atoms, double Sigma, double Epsilon) {
    double total_energy = 0.0;
    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = i + 1; j < atoms.size(); ++j) {
            double distance = Calculate_Distance(atoms[i].coordinates, atoms[j].coordinates);
            if (distance > 0) {
                total_energy += Calculate_LJ(distance, Sigma, Epsilon);
            }
        }
    }
    return total_energy;
}

double Compute_Second_Derivative(vector<Atom> &atoms, int atom1, int coord1, int atom2, int coord2, double delta, double Sigma, double Epsilon) {
    atoms[atom1].coordinates[coord1] += delta;
    atoms[atom2].coordinates[coord2] += delta;
    double energy_pp = Calculate_Total_Energy(atoms, Sigma, Epsilon);

    atoms[atom2].coordinates[coord2] -= 2 * delta;
    double energy_pm = Calculate_Total_Energy(atoms, Sigma, Epsilon);

    atoms[atom1].coordinates[coord1] -= 2 * delta;
    double energy_mm = Calculate_Total_Energy(atoms, Sigma, Epsilon);

    atoms[atom2].coordinates[coord2] += 2 * delta;
    double energy_mp = Calculate_Total_Energy(atoms, Sigma, Epsilon);

    atoms[atom1].coordinates[coord1] += delta;
    atoms[atom2].coordinates[coord2] -= delta;

    return (energy_pp - energy_pm - energy_mp + energy_mm) / (4 * delta * delta);
}

// Function to compute the Hessian matrix
Eigen::MatrixXd Compute_Hessian_Matrix(vector<Atom> &atoms, double delta, double Sigma, double Epsilon) {
    int num_atoms = atoms.size();
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(3 * num_atoms, 3 * num_atoms);

    for (int i = 0; i < num_atoms; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < num_atoms; ++k) {
                for (int l = 0; l < 3; ++l) {
                    double second_derivative = Compute_Second_Derivative(atoms, i, j, k, l, delta, Sigma, Epsilon);
                    hessian(3 * i + j, 3 * k + l) = second_derivative;
                    hessian(3 * k + l, 3 * i + j) = second_derivative; // Symmetric
                }
            }
        }
    }
    return hessian;
}

// Function to transform Hessian to mass-weighted coordinates
Eigen::MatrixXd TransformToMassWeighted(const Eigen::MatrixXd &hessian, const vector<Atom> &atoms) {
    int num_atoms = atoms.size();
    Eigen::MatrixXd mass_weighted_hessian = Eigen::MatrixXd::Zero(3 * num_atoms, 3 * num_atoms);

    for (int i = 0; i < 3 * num_atoms; ++i) {
        for (int j = 0; j < 3 * num_atoms; ++j) {
            int atom_i = i / 3;  // Determine which atom the row belongs to
            int atom_j = j / 3;  // Determine which atom the column belongs to
            double mass_factor = sqrt(atoms[atom_i].mass * atoms[atom_j].mass);
            mass_weighted_hessian(i, j) = hessian(i, j) / mass_factor;
        }
    }
    return mass_weighted_hessian;
}

// Function to compute vibrational frequencies from the mass-weighted Hessian
Eigen::VectorXd Compute_Vibrational_Frequencies(const Eigen::MatrixXd &mass_weighted_hessian) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mass_weighted_hessian);
    Eigen::VectorXd eigenvalues = solver.eigenvalues(); // Eigenvalues correspond to vibrational frequencies squared

    Eigen::VectorXd frequencies(eigenvalues.size());
    for (int i = 0; i < frequencies.size(); ++i) {
        frequencies(i) = (eigenvalues(i) > 0) ? sqrt(eigenvalues(i)) : 0.0; // Only positive eigenvalues are meaningful
    }
    return frequencies;
}

// Function to read input files
vector<Atom> ReadInput(const string &file_name) {
    ifstream infile(file_name);
    if (!infile.is_open()) {
        throw runtime_error("ERROR: Cannot open file: " + file_name);
    }

    vector<Atom> atoms;
    int num_atoms, charge;
    infile >> num_atoms >> charge;

    for (int i = 0; i < num_atoms; ++i) {
        int atomic_num;
        double x, y, z;
        infile >> atomic_num >> x >> y >> z;

        // Assign approximate masses based on atomic number
        double mass = (atomic_num == 1) ? 1.008 : (atomic_num == 8) ? 15.999 : 0.0;

        atoms.push_back({mass, {x, y, z}});
    }

    infile.close();
    return atoms;
}

// Main function
int main() {
    vector<string> input_files = {"../input_files/H2.txt", "../input_files/H2O.txt", "../input_files/HO.txt"};
    vector<string> output_files = {"../outputs/H2_results.txt", "../outputs/H2O_results.txt", "../outputs/HO_results.txt"};

    struct MoleculeParameters {
        double Sigma;
        double Epsilon;
        double delta;
    };

    vector<MoleculeParameters> molecule_params = {
        {2.75, 0.0104, 0.001},  // H2
        {3.16, 0.650, 0.001},   // H2O
        {3.46, 0.138, 0.001}    // HO
    };

    for (size_t i = 0; i < input_files.size(); ++i) {
        try {
            cout << "Processing " << input_files[i] << "..." << endl;

            vector<Atom> atoms = ReadInput(input_files[i]);
            const MoleculeParameters &params = molecule_params[i];

            Eigen::MatrixXd hessian = Compute_Hessian_Matrix(atoms, params.delta, params.Sigma, params.Epsilon);
            Eigen::MatrixXd mass_weighted_hessian = TransformToMassWeighted(hessian, atoms);

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mass_weighted_hessian);
            Eigen::VectorXd eigenvalues = solver.eigenvalues();
            Eigen::MatrixXd eigenvectors = solver.eigenvectors();

            ofstream outfile(output_files[i]);
            if (!outfile.is_open()) throw runtime_error("ERROR: Cannot open file: " + output_files[i]);

            outfile << "Hessian Matrix:\n" << hessian << "\n\n";
            outfile << "Mass-Weighted Hessian Matrix:\n" << mass_weighted_hessian << "\n\n";
            outfile << "Eigenvalues (Vibrational Frequencies Squared):\n" << eigenvalues << "\n\n";
            outfile << "Eigenvectors (Normal Modes):\n" << eigenvectors << "\n";

            outfile.close();
            cout << "Results written to " << output_files[i] << endl;
        } catch (const runtime_error &e) {
            cerr << e.what() << endl;
        }
    }

    return 0;
}
