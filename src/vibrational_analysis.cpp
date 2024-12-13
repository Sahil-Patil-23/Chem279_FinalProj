#include <iostream>           
#include <fstream>            
#include <cmath>              
#include <vector>             
#include <Eigen/Dense>        
using namespace std;          

// Struct to store atomic data: mass, charge, and coordinates
struct Atom {
    double mass;                 // Atomic mass in atomic mass units (amu)
    double charge;               // Partial charge for dipole derivative calculations
    vector<double> coordinates;  // x, y, z coordinates in Ångstrom or Bohr
};

// Function to calculate the distance between two atoms
double Calculate_Distance(const vector<double> &atom1, const vector<double> &atom2) {
    // Computes the Euclidean distance between two points in 3D space
    return sqrt(pow(atom1[0] - atom2[0], 2) +
                pow(atom1[1] - atom2[1], 2) +
                pow(atom1[2] - atom2[2], 2));
}

// Function to calculate the Lennard-Jones potential between two atoms
double Calculate_LJ(double distance, double Sigma, double Epsilon) {
    // Calculates the Lennard-Jones potential using the given parameters
    double sig_dist = Sigma / distance;       // Ratio of sigma to distance
    double term6 = pow(sig_dist, 6);          // (Sigma / distance)^6
    double term12 = pow(sig_dist, 12);        // (Sigma / distance)^12
    return 4 * Epsilon * (term12 - term6);    // LJ potential formula
}

// Function to calculate the total energy of a cluster of atoms
double Calculate_Total_Energy(const vector<Atom> &atoms, double Sigma, double Epsilon) {
    double total_energy = 0.0;                // Initialize total energy
    for (size_t i = 0; i < atoms.size(); ++i) {  // Loop through all pairs of atoms
        for (size_t j = i + 1; j < atoms.size(); ++j) {
            double distance = Calculate_Distance(atoms[i].coordinates, atoms[j].coordinates);
            if (distance > 0) {               // Avoid division by zero
                total_energy += Calculate_LJ(distance, Sigma, Epsilon); // Add LJ potential
            }
        }
    }
    return total_energy;                      // Return total potential energy
}

// Function to compute second derivatives for the Hessian matrix
double Compute_Second_Derivative(vector<Atom> &atoms, int atom1, int coord1, int atom2, int coord2, double delta, double Sigma, double Epsilon) {
    // Perturb atom1 and atom2 coordinates in positive and negative directions
    atoms[atom1].coordinates[coord1] += delta;
    atoms[atom2].coordinates[coord2] += delta;
    double energy_pp = Calculate_Total_Energy(atoms, Sigma, Epsilon); // Energy (++)

    atoms[atom2].coordinates[coord2] -= 2 * delta;
    double energy_pm = Calculate_Total_Energy(atoms, Sigma, Epsilon); // Energy (+-)

    atoms[atom1].coordinates[coord1] -= 2 * delta;
    double energy_mm = Calculate_Total_Energy(atoms, Sigma, Epsilon); // Energy (--)

    atoms[atom2].coordinates[coord2] += 2 * delta;
    double energy_mp = Calculate_Total_Energy(atoms, Sigma, Epsilon); // Energy (-+)

    // Restore original positions
    atoms[atom1].coordinates[coord1] += delta;
    atoms[atom2].coordinates[coord2] -= delta;

    // Compute second derivative using finite difference formula
    return (energy_pp - energy_pm - energy_mp + energy_mm) / (4 * delta * delta);
}

// Function to compute the Hessian matrix
Eigen::MatrixXd Compute_Hessian_Matrix(vector<Atom> &atoms, double delta, double Sigma, double Epsilon) {
    int num_atoms = atoms.size();            // Number of atoms in the molecule
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(3 * num_atoms, 3 * num_atoms); // Initialize Hessian matrix

    // Loop over all atoms and their x, y, z coordinates
    for (int i = 0; i < num_atoms; ++i) {
        for (int j = 0; j < 3; ++j) {        // Loop over x, y, z of atom i
            for (int k = 0; k < num_atoms; ++k) {
                for (int l = 0; l < 3; ++l) { // Loop over x, y, z of atom k
                    // Compute second derivative for Hessian
                    double second_derivative = Compute_Second_Derivative(atoms, i, j, k, l, delta, Sigma, Epsilon);
                    hessian(3 * i + j, 3 * k + l) = second_derivative; // Update Hessian
                }
            }
        }
    }
    return hessian;                          // Return the Hessian matrix
}

// Function to transform Hessian to mass-weighted coordinates
Eigen::MatrixXd TransformToMassWeighted(const Eigen::MatrixXd &hessian, const vector<Atom> &atoms) {
    int num_atoms = atoms.size();            // Number of atoms
    Eigen::MatrixXd mass_weighted_hessian = Eigen::MatrixXd::Zero(3 * num_atoms, 3 * num_atoms);

    // Loop over the Hessian matrix to apply mass-weighting
    for (int i = 0; i < 3 * num_atoms; ++i) {
        for (int j = 0; j < 3 * num_atoms; ++j) {
            int atom_i = i / 3;              // Atom corresponding to row i
            int atom_j = j / 3;              // Atom corresponding to column j
            double mass_factor = sqrt(atoms[atom_i].mass * atoms[atom_j].mass); // Mass factor
            mass_weighted_hessian(i, j) = hessian(i, j) / mass_factor; // Apply mass weighting
        }
    }
    return mass_weighted_hessian;            // Return the mass-weighted Hessian
}

// Function to determine if vibrational modes are IR-active
bool Is_IR_Active(const Eigen::VectorXd &eigenvector, const vector<Atom> &atoms) {
    // Initialize a vector to store the total dipole derivative
    Eigen::Vector3d dipole_derivative = Eigen::Vector3d::Zero();

    // Check for homonuclear diatomic molecules, which are always IR-inactive
    if (atoms.size() == 2 && atoms[0].charge == atoms[1].charge) {
        return false;
    }

    // Iterate over each atom in the molecule
    for (int i = 0; i < atoms.size(); ++i) {
        // Extract the x, y, and z components of the displacement vector for the current atom
        Eigen::Vector3d displacement(eigenvector(3 * i), eigenvector(3 * i + 1), eigenvector(3 * i + 2));

        // Calculate the contribution of the atom's charge and displacement to the total dipole derivative
        dipole_derivative += atoms[i].charge * displacement;
    }

    // Check if the magnitude of the total dipole derivative is significant
    return dipole_derivative.norm() > 1e-6;
}

// Function to read input files and assign partial charges/masses for C, H, O, and N
vector<Atom> ReadInput(const string &file_name) {
    ifstream infile(file_name);              // Open input file
    if (!infile.is_open()) {
        throw runtime_error("ERROR: Cannot open file: " + file_name); // Handle file error
    }

    vector<Atom> atoms;                      // Initialize vector to store atoms
    int num_atoms, charge;                   // Number of atoms and total charge of molecule
    infile >> num_atoms >> charge;           // Read first line of input file

    for (int i = 0; i < num_atoms; ++i) {
        int atomic_num;
        double x, y, z;
        infile >> atomic_num >> x >> y >> z; // Read atomic number and coordinates

        // Assign masses and partial charges based on atomic number
        double mass = (atomic_num == 1) ? 1.008 :  // Hydrogen
                      (atomic_num == 8) ? 15.999 : // Oxygen
                      (atomic_num == 6) ? 12.011 : // Carbon
                      (atomic_num == 7) ? 14.007 : // Nitrogen
                      0.0;                         // Default

        double charge = (atomic_num == 1) ? 0.42 :  // Partial charge for Hydrogen
                        (atomic_num == 8) ? -0.84 : // Partial charge for Oxygen
                        (atomic_num == 6) ? 0.0 :   // Partial charge for Carbon
                        (atomic_num == 7) ? 0.0 :   // Partial charge for Nitrogen
                        0.0;                        // Default

        atoms.push_back({mass, charge, {x, y, z}}); // Add atom to list
    }

    infile.close();                         // Close input file
    return atoms;                           // Return the list of atoms
}

// Function to convert coordinates from Ångstrom to Bohr
void Convert_To_Bohr(vector<Atom> &atoms) {
    const double angstrom_to_bohr = 1.8897259886;   // Conversion factor
    for (Atom &atom : atoms) {
        for (double &coord : atom.coordinates) {
            coord *= angstrom_to_bohr;             // Convert each coordinate
        }
    }
}

// Main function
int main() {
    // Input/output file paths for different molecules
    vector<string> input_files = {"../input_files/H2.txt", "../input_files/H2O.txt", "../input_files/NH3.txt"};
    vector<string> output_files = {"../outputs/H2_results.txt", "../outputs/H2O_results.txt", "../outputs/NH3_results.txt"};

    struct MoleculeParameters {
        double Sigma;
        double Epsilon;
        double delta;
    };

    // Define molecule-specific parameters (sigma, epsilon, and delta values)
    // https://basicmedicalkey.com/liquids-liquid-crystals-and-ionic-liquids/
    vector<MoleculeParameters> molecule_params = {
        {2.827, 0.0597, 1e-6},  // H2
        {2.641, 0.0698, 1e-6},  // H2O
        {2.900, 0.0481, 1e-6},  // NH3
        {3.758, 0.0128, 1e-6}   // CH4
    };

    // Iterate over each molecule
    for (size_t i = 0; i < input_files.size(); ++i) {
        try {
            cout << "Processing " << input_files[i] << "..." << endl;

            // Read atomic data from input file
            vector<Atom> atoms = ReadInput(input_files[i]);

            // Convert coordinates to Bohr
            Convert_To_Bohr(atoms);

            // Retrieve molecule parameters
            const MoleculeParameters &params = molecule_params[i];

            // Compute Hessian matrix
            Eigen::MatrixXd hessian = Compute_Hessian_Matrix(atoms, params.delta, params.Sigma, params.Epsilon);

            // Transform to mass-weighted Hessian
            Eigen::MatrixXd mass_weighted_hessian = TransformToMassWeighted(hessian, atoms);

            // Diagonalize the mass-weighted Hessian
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mass_weighted_hessian);
            Eigen::VectorXd eigenvalues = solver.eigenvalues();   // Vibrational frequencies squared
            Eigen::MatrixXd eigenvectors = solver.eigenvectors(); // Normal modes

            // Set small negative eigenvalues to zero to avoid numerical noise
            for (int j = 0; j < eigenvalues.size(); ++j) {
                if (eigenvalues[j] < 0) eigenvalues[j] = 0.0;
            }

            // Convert eigenvalues to vibrational frequencies (cm^-1)
            const double conversion_factor = 5140.486;            // Bohr to cm^-1
            Eigen::VectorXd frequencies_cm(eigenvalues.size());
            for (int j = 0; j < eigenvalues.size(); ++j) {
                frequencies_cm[j] = sqrt(eigenvalues[j]) * conversion_factor;
            }

            // Determine IR activity for each mode
            vector<bool> ir_active(eigenvectors.cols());
            for (int j = 0; j < eigenvectors.cols(); ++j) {
                ir_active[j] = Is_IR_Active(eigenvectors.col(j), atoms);
                Eigen::Vector3d dipole_derivative = Eigen::Vector3d::Zero();
                for (int k = 0; k < atoms.size(); ++k) {
                    Eigen::Vector3d displacement(eigenvectors(3 * k, j), eigenvectors(3 * k + 1, j), eigenvectors(3 * k + 2, j));
                    dipole_derivative += atoms[k].charge * displacement;
                }
            }

            // Write results to output file
            ofstream outfile(output_files[i]);
            if (!outfile.is_open()) throw runtime_error("ERROR: Cannot open file: " + output_files[i]);

            outfile << "Hessian Matrix:\n" << hessian << "\n\n";
            outfile << "Mass-Weighted Hessian Matrix:\n" << mass_weighted_hessian << "\n\n";
            outfile << "Eigenvalues (Vibrational Frequencies Squared):\n" << eigenvalues << "\n\n";
            outfile << "Vibrational Frequencies (cm^-1):\n" << frequencies_cm << "\n\n";
            outfile << "Eigenvectors (Normal Modes):\n" << eigenvectors << "\n\n";
            outfile << "IR Activity (1 = Active, 0 = Inactive):\n";
            for (bool active : ir_active) {
                outfile << (active ? "1" : "0") << "\n";
            }

            outfile.close();
            cout << "Results written to " << output_files[i] << endl;
        } catch (const runtime_error &e) {
            cerr << e.what() << endl;
        }
    }

    return 0;
}

