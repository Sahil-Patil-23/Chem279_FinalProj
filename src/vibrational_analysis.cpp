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

// Function to transform Hessian to mass-weighted coordinates
Eigen::MatrixXd TransformToMassWeighted(const Eigen::MatrixXd &hessian, const std::vector<Atom> &atoms) {
    int num_atoms = atoms.size();
    Eigen::MatrixXd mass_weighted_hessian = Eigen::MatrixXd::Zero(3 * num_atoms, 3 * num_atoms);

    for (int i = 0; i < 3 * num_atoms; ++i) {
        for (int j = 0; j < 3 * num_atoms; ++j) {
            int atom_i = i / 3;  // Determine which atom the row coordinate belongs to
            int atom_j = j / 3;  // Determine which atom the column coordinate belongs to
            mass_weighted_hessian(i, j) = hessian(i, j) / sqrt(atoms[atom_i].mass * atoms[atom_j].mass);
        }
    }

    return mass_weighted_hessian;
}

// Function to compute vibrational frequencies from mass-weighted Hessian Matrix
Eigen::VectorXd Compute_Vibrational_Frequencies(const Eigen::MatrixXd &mass_weighted_hessian){

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mass_weighted_hessian); // Computes eigenvalues 
    Eigen::VectorXd Eigen_vals = solver.eigenvalues(); // Retrieves eigenvalues of the mass-weighted hessian

    Eigen::VectorXd frequencies(Eigen_vals.size());   
    for(int i = 0; i < frequencies.size(); i++){
        frequencies(i) = Eigen_vals(i) > 0 ? sqrt(Eigen_vals(i)) : 0.0; // Eigenvalues are converted to frequencies by taking the square root. Only positive values are of any meaning
    } 

    return frequencies;
}

// Function to calculate teh Partition function
double Calculate_Partition_Function(const Eigen::VectorXd& frequencies, double temperature) {
    double partition_function = 1.0;
    const double k_B = 1.38e-23;  // Boltzmann constant in J/K
    const double h = 6.626e-34;   // Planck constant in J·s
    const double c = 3.00e10;     // Speed of light in cm/s

    for (int i = 0; i < frequencies.size(); i++) {
        if (frequencies(i) > 0) {
            double energy = h * c * frequencies(i); // Convert to energy in Joules
            partition_function *= 1.0 / (1.0 - exp(-energy / (k_B * temperature)));
        }
    }
    return partition_function;
}

// Function that reads input files
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
        atoms.push_back({static_cast<double>(atomic_num), {x, y, z}});
    }

    infile.close();
    return atoms;
}

// Function that generates output files
void WriteOutput(const string &file_name, const Eigen::MatrixXd &hessian) {
    ofstream outfile(file_name);
    if (!outfile.is_open()) {
        throw runtime_error("ERROR: Cannot open file: " + file_name);
    }

    outfile << "Hessian Matrix:" << endl;
    outfile << hessian << endl;
    outfile.close();
}

int main(){

    vector<string> input_files = {"../input_files/H2.txt", "../input_files/H2O.txt", "../input_files/HO.txt"};
    vector<string> output_files = {"../outputs/H2_results.txt", "../outputs/H2O_results.txt", "../outputs/HO_results.txt"};

    // For correct sigma/epsilon/delta values based on input molecule
    struct MoleculeParameters {
        double Sigma;
        double Epsilon;
        double delta;
    };

    vector<MoleculeParameters> molecule_params = {
        {2.75, 0.0104, 0.001},  // H2: Sigma (Å), Epsilon (eV), delta
        {3.16, 0.650, 0.001},   // H2O: Sigma (Å), Epsilon (eV), delta
        {3.46, 0.138, 0.001}    // HO: Sigma (Å), Epsilon (eV), delta
    };

    for (size_t i = 0; i < input_files.size(); ++i) {
        try {
            std::cout << "Processing " << input_files[i] << "..." << std::endl;

            // Read atoms from the input file
            std::vector<Atom> atoms = ReadInput(input_files[i]);

            // Retrieve molecule-specific parameters
            const MoleculeParameters &params = molecule_params[i];

            // Compute the Hessian matrix
            Eigen::MatrixXd hessian = Compute_Hessian_Matrix(atoms, params.delta, params.Sigma, params.Epsilon);

            // Transform Hessian to mass-weighted coordinates
            Eigen::MatrixXd mass_weighted_hessian = TransformToMassWeighted(hessian, atoms);

            // Diagonalize the mass-weighted Hessian
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mass_weighted_hessian);
            Eigen::VectorXd eigenvalues = solver.eigenvalues();   // Vibrational frequencies squared
            Eigen::MatrixXd eigenvectors = solver.eigenvectors(); // Normal modes

            // Output the results
            std::ofstream outfile(output_files[i]);
            if (!outfile.is_open()) {
                throw std::runtime_error("ERROR: Cannot open file: " + output_files[i]);
            }

            // Write the eigenvalues (vibrational frequencies squared) and eigenvectors (normal modes) to the file
            outfile << "Mass-Weighted Hessian Matrix:\n" << mass_weighted_hessian << "\n\n";
            outfile << "Eigenvalues (vibrational frequencies squared):\n" << eigenvalues << "\n\n";
            outfile << "Eigenvectors (normal modes):\n" << eigenvectors << "\n";
            outfile.close();

            std::cout << "Results written to " << output_files[i] << std::endl;
        } catch (const std::runtime_error &e) {
            std::cerr << e.what() << std::endl;
        }
    }

    return 0;
}