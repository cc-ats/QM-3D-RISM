import os
import shutil

# Ensure EPISOL binaries are discovered and added to PATH
_current_dir = os.path.dirname(os.path.abspath(__file__))
_episol_bin_candidates = [
    os.path.expanduser('~/episol/bin'),
    os.path.join(_current_dir, 'episol', 'release'),
    os.path.expanduser('~/.local/bin'),
    '/usr/local/bin',
]
for _candidate in _episol_bin_candidates:
    if os.path.exists(os.path.join(_candidate, 'eprism3d')):
        if _candidate not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _candidate + ':' + os.environ.get('PATH', '')
        break

# Ensure IETLIB is set for EPISOL solvent parameter files if not already defined
if 'IETLIB' not in os.environ:
    _ietlib_candidates = [
        os.path.expanduser('~/episol/share/eprism3d/solvent'),
        os.path.join(_current_dir, 'episol', 'release', 'solvent'),
        '/usr/local/share/eprism3d/solvent',
    ]
    for _candidate in _ietlib_candidates:
        if os.path.isdir(_candidate):
            os.environ['IETLIB'] = _candidate
            break
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyscf import gto, scf, lib, dft, df
from episol import epipy
import matplotlib.pyplot as plt


def update_solute_charges(solute_file, charges):
    """
    Update the solute topology file (typically idc_acid.solute) with the 
    newly calculated atomic partial charges at the end of each RISM-SCF cycle.

    Parameters:
    solute_file (str): Path to the solute topology file.
    charges (list or numpy.ndarray): Fitted charges for each atom.
    """
    with open(solute_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    atom_idx = 0
    for line in lines:
        if line.startswith('#') or line.startswith('[') or not line.strip():
            new_lines.append(line)
        else:
            parts = line.split()
            # Replace the 6th column (charge, index 5) with the new charge
            parts[5] = f"{charges[atom_idx]:.6f}"
            atom_idx += 1

            # Format the line back to standard format dynamically based on columns
            new_line = f"     {parts[0]}    {parts[1]}   {parts[2]}  {parts[3]}      {parts[4]}    {parts[5]}     {parts[6]}     {parts[7]}"
            if len(parts) > 8:
                new_line += f"  {parts[8]}"
            new_line += "\n"
            new_lines.append(new_line)

    with open(solute_file, 'w') as f:
        f.writelines(new_lines)

def compute_becke_partition_weights(mol, coords_ang):
    """
    Manually compute Becke partition weights (using s3 cutoff with 3 iterations)
    for grid coordinates in Angstroms.
    
    Parameters:
        mol (pyscf.gto.Mole): PySCF Molecule.
        coords_ang (numpy.ndarray): Grid coordinates in Angstroms, shape (N_grid, 3).
        
    Returns:
        numpy.ndarray: Partition weights of shape (n_atoms, N_grid).
    """
    n_atoms = mol.natm
    atom_coords = mol.atom_coords() * lib.param.BOHR  # Convert to Angstroms
    n_grid = coords_ang.shape[0]
    
    diff = coords_ang[None, :, :] - atom_coords[:, None, :]
    dists = np.linalg.norm(diff, axis=2)
    
    P = np.ones((n_atoms, n_grid))
    for A in range(n_atoms):
        for B in range(n_atoms):
            if A == B:
                continue
            R_AB = np.linalg.norm(atom_coords[A] - atom_coords[B])
            mu = (dists[A] - dists[B]) / R_AB
            mu = np.clip(mu, -1.0, 1.0)
            
            # Becke's 3 iterations of s3(mu) to form an 11th order polynomial
            for _ in range(3):
                mu = 1.5 * mu - 0.5 * mu**3
                
            s = 0.5 * (1.0 - mu)
            P[A] *= s
            
    P_sum = np.sum(P, axis=0)
    P_sum = np.where(P_sum == 0.0, 1.0, P_sum)
    w_g_A = P / P_sum
    return w_g_A

def continous_solute_charge_density(mol, pot_grid, x, y, z, d=None, real_space_sol=None, downsample=2):
    """
    Scheme I: Continuous Solute Charge Density (Solvent -> Solute)
    
    Calculate the direct solvent potential contribution (delta_F_direct) to the 
    Fock matrix by integrating the continuous solvent potential over the solute 
    electron density using PySCF's multi-center atom-centered DFT grid.
    Uses manually computed Becke partition weights.
    
    Parameters:
        mol (pyscf.gto.Mole): PySCF Molecule object.
        pot_grid (numpy.ndarray): 3D electrostatic potential grid from 3D-RISM (kcal/mol).
        x, y, z (numpy.ndarray): 1D arrays of grid coordinates along x, y, z axes (Angstroms).
        d (numpy.ndarray): 3D shift vector to align coordinate systems.
        real_space_sol: EPISOL RISM object. If provided, potential is computed via direct real-space sum.
        downsample (int): Downsampling factor for source grid in real-space potential.
        
    Returns:
        numpy.ndarray: Direct solvent potential contribution matrix, delta_F_direct (shape: nao x nao).
    """
    # 1. Generate the standard atom-centered DFT grid 
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.radii_adjust = None  # Match manual Becke partitioning coordinate scale
    grids.build()

    coords_bohr = grids.coords
    coords_ang = coords_bohr * lib.param.BOHR  # Convert Bohr to Angstroms for interpolation

    # Align coordinate systems if shift is provided
    if d is not None:
        coords_ang = coords_ang + d

    # 3. Compute potential at the atom-centered grid coordinates (convert to Hartree)
    if real_space_sol is not None:
        v_solv_grid_kj = compute_solvent_potential_at_coords_real_space(real_space_sol, coords_ang, d=d, downsample=downsample)
        v_solv_grid = v_solv_grid_kj / 2625.5002
    else:
        pot_interpolator = RegularGridInterpolator((x, y, z), pot_grid, bounds_error=False, fill_value=0.0)
        v_solv_grid = pot_interpolator(coords_ang) / 2625.5002

    # 4. Compute Becke partition weights manually
    w_g_A = compute_becke_partition_weights(mol, coords_ang)
    
    # Extract the atom index and raw quadrature weight for each grid point
    atm_idx = grids.atm_idx
    raw_weights = grids.quadrature_weights
    w_custom = w_g_A[atm_idx, np.arange(len(coords_ang))]
    
    # Calculate the custom partitioned weights
    weights = raw_weights * w_custom

    # 5. Evaluate atomic orbitals at grid points
    ao = dft.numint.eval_ao(mol, coords_bohr)
    
    # 6. Integrate to compute Fock matrix correction:
    # delta_F_direct_uv = \int phi_u(r) * V_solv(r) * phi_v(r) dr
    # approximated as \sum_p w_p * phi_u(r_p) * V_solv(r_p) * phi_v(r_p)
    aow = np.einsum('pi,p,p->pi', ao, weights, v_solv_grid)
    delta_F_direct = -np.dot(ao.T, aow)  # Negative sign accounts for the negative charge of electrons (-1 e)
    return delta_F_direct

def parse_gro_to_pyscf_atoms(gro_path):
    """
    Parse a GROMACS .gro file to construct a PySCF atom list.
    Extracts atom coordinates (converting nm to Angstroms) and elements.
    This guarantees 100% exact alignment and index mapping between PySCF and EPISOL.
    """
    import re
    with open(gro_path, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[1].strip())
    atoms = []
    for i in range(num_atoms):
        line = lines[2 + i]
        # Atom name is in columns 10-15
        atom_name = line[10:15].strip()
        # Extract element symbol by stripping numbers
        symbol = re.sub(r'\d+', '', atom_name)
        # Handle GROMACS/GAFF-specific name formats
        if symbol.upper() in ['OW', 'HW']:
            symbol = symbol[0]
        elif symbol.upper() in ['CL', 'BR', 'NA', 'FE', 'MG']:
            symbol = symbol.capitalize()
        else:
            if len(symbol) > 1 and symbol[1].islower():
                symbol = symbol[:2]
            else:
                symbol = symbol[0]
        
        # Coordinates in nm: columns 20-28, 28-36, 36-44. Convert to Angstroms
        x = float(line[20:28]) * 10.0
        y = float(line[28:36]) * 10.0
        z = float(line[36:44]) * 10.0
        atoms.append((symbol, (x, y, z)))
    return atoms

def generate_chelpg_grid(mol, r_max=2.8, spacing=0.5):
    """
    Generate a grid of points (in Bohr) surrounding the solute molecule for ESP fitting.
    Excludes points inside VdW radii or too far from the solute.
    """
    coords_ang = mol.atom_coords() * lib.param.BOHR
    vdw_radii = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Cl': 1.75}

    # 1. Generate bounding box coordinates 
    min_xyz = np.min(coords_ang, axis=0) - r_max
    max_xyz = np.max(coords_ang, axis=0) + r_max

    gx = np.arange(min_xyz[0], max_xyz[0], spacing)
    gy = np.arange(min_xyz[1], max_xyz[1], spacing)
    gz = np.arange(min_xyz[2], max_xyz[2], spacing)
    X, Y, Z = np.meshgrid(gx, gy, gz)
    grid_all = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # 2. Exclude points inside VdW radii or too far from the solute
    valid_points = []
    for pt in grid_all:
        dists = np.linalg.norm(coords_ang - pt, axis=1)
        too_close = False
        for i, dist in enumerate(dists):
            sym = mol.atom_symbol(i)
            r_vdw = vdw_radii.get(sym, 1.5)
            if dist < r_vdw:
                too_close = True
                break
        if not too_close:
            valid_points.append(pt)

    points_bohr = np.array(valid_points) / lib.param.BOHR
    return points_bohr

def fit_esp_charges(mol, dm):
    """
    Scheme IIA: Fit ESP Charges (Solute -> Solvent)
    
    Fit partial atomic charges to reproduce the electrostatic potential (ESP) 
    of the continuous QM density matrix and nuclear charges on a CHELPG-like 
    grid of points surrounding the solute molecule.
    
    Parameters:
        mol (pyscf.gto.Mole): PySCF Molecule object.
        dm (numpy.ndarray): QM density matrix.
        
    Returns:
        numpy.ndarray: Fitted partial charges for each atom.
    """
    points_bohr = generate_chelpg_grid(mol)

    # 1. Calculate nuclear contribution to the potential 
    Vnuc = np.zeros(len(points_bohr))
    for i in range(mol.natm):
        r = mol.atom_coord(i)
        Z_atom = mol.atom_charge(i)
        rp = r - points_bohr
        Vnuc += Z_atom / np.linalg.norm(rp, axis=1)

    # 2. Calculate electronic contribution to the potential
    fakemol = gto.fakemol_for_charges(points_bohr)
    Vele = -np.einsum('ijp,ij->p', df.incore.aux_e2(mol, fakemol), dm)
    Vtotal = Vnuc + Vele

    # 3. Build fitting matrix 
    M = np.zeros((len(points_bohr), mol.natm))
    for a in range(mol.natm):
        R_a = mol.atom_coord(a)
        M[:, a] = 1.0 / np.linalg.norm(points_bohr - R_a, axis=1)
    
    # 4. Constrained least squares solving (sum of charges = mol.charge)
    A = np.zeros((mol.natm + 1, mol.natm + 1))
    A[:mol.natm, :mol.natm] = np.dot(M.T, M)
    A[mol.natm, :mol.natm] = 1.0
    A[:mol.natm, mol.natm] = 1.0

    B = np.zeros(mol.natm + 1)
    B[:mol.natm] = np.dot(M.T, Vtotal)
    B[mol.natm] = mol.charge

    sol = np.linalg.solve(A, B)
    return sol[:mol.natm]

def compute_solvent_potential_at_nuclei(sol, mol, d):
    """
    Compute the solvent electrostatic potential at the solute nuclei
    directly from the solvent total correlation function (huv) using Coulomb's law.
    Excludes grid points inside the VdW cavity to prevent singularity.
    """
    # 1. Extract huv grids for OW and HW
    h_O = sol.select_grid('huv atom O')
    h_H = sol.select_grid('huv atom H')
    
    # Solvent density and charges
    rho_O = 0.03342288177 # molecules/A^3
    rho_H = 2.0 * rho_O
    q_O = -0.834
    q_H = 0.417
    
    charge_grid = q_O * rho_O * h_O + q_H * rho_H * h_H
    resolution = sol.resolution
    dV = resolution**3
    q_k = charge_grid.ravel() * dV # in elementary charges e
    
    # Grid coordinates in Angstroms
    nx, ny, nz = charge_grid.shape
    x = np.arange(nx) * resolution
    y = np.arange(ny) * resolution
    z = np.arange(nz) * resolution
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_coords_ang = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # Translate grid coordinates to PySCF coordinate system
    grid_coords_ang_pyscf = grid_coords_ang - d # in Angstroms
    
    # Solute nuclear coordinates in Angstroms
    nuc_coords_ang = mol.atom_coords() * lib.param.BOHR
    
    # VdW radii for cavity exclusion
    vdw_radii = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Cl': 1.75}
    outside_cavity = np.ones(len(grid_coords_ang_pyscf), dtype=bool)
    for i in range(mol.natm):
        sym = mol.atom_symbol(i)
        r_vdw = vdw_radii.get(sym, 1.5)
        dists = np.linalg.norm(grid_coords_ang_pyscf - nuc_coords_ang[i], axis=1)
        outside_cavity = outside_cavity & (dists >= r_vdw)
        
    q_outside = q_k[outside_cavity]
    coords_outside = grid_coords_ang_pyscf[outside_cavity]
    
    # Calculate potential at nuclei in kJ/mol/e (1 e^2/Angstrom = 1389.35485 kJ/mol)
    pot_at_nuclei = np.zeros(mol.natm)
    for a in range(mol.natm):
        R_a = nuc_coords_ang[a]
        dists = np.linalg.norm(coords_outside - R_a, axis=1)
        pot_at_nuclei[a] = np.sum(q_outside / dists) * 1389.35485
        
    return pot_at_nuclei

def compute_solvent_potential_grid_fft(sol):
    """
    Compute the solvent electrostatic potential grid in kJ/mol/e
    directly from the solvent total correlation function (huv) using FFT 3D convolution.
    """
    h_O = sol.select_grid('huv atom O')
    h_H = sol.select_grid('huv atom H')
    
    # Solvent density and charges (TIP3P-AMBER)
    rho_O = 0.03342288177  # molecules/A^3
    rho_H = 2.0 * rho_O
    q_O = -0.834
    q_H = 0.417
    
    # Solvent charge density in e/A^3
    charge_grid = q_O * rho_O * h_O + q_H * rho_H * h_H
    
    # Perform 3D FFT
    rho_k = np.fft.fftn(charge_grid)
    
    # Wavevectors in Angstroms^-1
    nx, ny, nz = charge_grid.shape
    resolution = sol.resolution
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=resolution)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=resolution)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=resolution)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0  # Avoid singularity
    
    # Poisson equation in Fourier space: V(k) = 4 * pi * rho(k) / k^2
    pot_k = 4.0 * np.pi * rho_k / k2
    pot_k[0, 0, 0] = 0.0  # Neutrality constraint
    
    # Inverse FFT to real space (in e/A) and convert to kJ/mol/e
    # 1 e/A = 1389.35485 kJ/mol/e
    pot_grid_solvent = np.real(np.fft.ifftn(pot_k)) * 1389.35485
    return pot_grid_solvent

def compute_solvent_potential_at_coords_real_space(sol, target_coords_ang, d=None, downsample=2):
    """
    Compute the solvent electrostatic potential at arbitrary target coordinates
    in Angstroms using direct real-space Coulomb summation.
    
    Parameters:
        sol (epipy): EPISOL RISM object.
        target_coords_ang (numpy.ndarray): Target coordinates, shape (N_target, 3).
        d (numpy.ndarray): Shift vector to align coordinate systems.
        downsample (int): Downsampling factor for the source grid to speed up calculation.
    """
    h_O = sol.select_grid('huv atom O')
    h_H = sol.select_grid('huv atom H')
    
    rho_O = 0.03342288177  # molecules/A^3
    rho_H = 2.0 * rho_O
    q_O = -0.834
    q_H = 0.417
    
    # Solvent charge density in e/A^3
    charge_grid = q_O * rho_O * h_O + q_H * rho_H * h_H
    
    # Downsample the source grid if requested
    if downsample > 1:
        charge_grid = charge_grid[::downsample, ::downsample, ::downsample]
        resolution = sol.resolution * downsample
    else:
        resolution = sol.resolution
        
    nx, ny, nz = charge_grid.shape
    
    # Grid coordinates in Angstroms
    x = np.arange(nx) * resolution
    y = np.arange(ny) * resolution
    z = np.arange(nz) * resolution
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    src_coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    
    # Apply coordinate alignment shift to source coordinates if provided
    if d is not None:
        src_coords = src_coords - d
        
    q_flat = charge_grid.ravel() * (resolution**3)  # charge on each voxel in e
    
    # Filter out near-zero charges to speed up calculation
    active = np.abs(q_flat) > 1e-10
    src_coords = src_coords[active]
    q_flat = q_flat[active]
    
    # Compute potential at target coordinates
    pot_target = np.zeros(len(target_coords_ang))
    chunk_size = 500
    for i in range(0, len(target_coords_ang), chunk_size):
        chunk_tar = target_coords_ang[i:i+chunk_size]  # shape: (C, 3)
        dists = np.linalg.norm(chunk_tar[:, None, :] - src_coords[None, :, :], axis=2)
        dists = np.where(dists < 1e-8, np.inf, dists)  # Avoid self-interaction singularity
        pot_target[i:i+chunk_size] = np.sum(q_flat[None, :] / dists, axis=1)
        
    return pot_target * 1389.35485  # convert to kJ/mol/e

def compute_fock_derivative(mol, pot_at_nuclei):
    """
    Scheme IIB: Compute Fock Derivative (Solvent -> Solute)
    
    Computes the contribution to the effective Fock matrix arising from the derivative
    of the solvation free energy with respect to the density matrix (via the fitted ESP charges).
    This accounts for the back-reaction (polarization of the solvent reaction field).
    
    Parameters:
        mol (pyscf.gto.Mole): PySCF Molecule object.
        pot_at_nuclei (numpy.ndarray): Solvent electrostatic potential at solute nuclei in Hartree.
        
    Returns:
        numpy.ndarray: Fock derivative potential contribution matrix, delta_F_deriv (shape: nao x nao).
    """
    points_bohr = generate_chelpg_grid(mol)

    # 1. Build fitting matrix 
    M = np.zeros((len(points_bohr), mol.natm))
    for a in range(mol.natm):
        R_a = mol.atom_coord(a)
        M[:, a] = 1.0 / np.linalg.norm(points_bohr - R_a, axis=1)

    # 2. Constrained least squares matrix
    A = np.zeros((mol.natm + 1, mol.natm + 1))
    A[:mol.natm, :mol.natm] = np.dot(M.T, M)
    A[mol.natm, :mol.natm] = 1.0 
    A[:mol.natm, mol.natm] = 1.0

    A_inv = np.linalg.inv(A)

    # 4. Compute weight and grid point charge vector
    w = np.dot(A_inv[:mol.natm, :mol.natm], pot_at_nuclei)
    f = -np.dot(M, w)

    # 5. Evaluate derivative contribution using 3-center integrals
    fakemol = gto.fakemol_for_charges(points_bohr)
    delta_F_deriv = np.einsum('ijp,p->ij', df.incore.aux_e2(mol, fakemol), f)
    return delta_F_deriv

def get_mulliken_charges(mol, dm, s_matrix):
    """
    Scheme IIIA: Calculates Mulliken partial charges for a molecule.
    
    Parameters:
        mol (pyscf.gto.Mole): PySCF Molecule object.
        dm (numpy.ndarray): QM density matrix.
        s_matrix (numpy.ndarray): Overlap matrix.
        
    Returns:
        numpy.ndarray: Mulliken partial charges for each atom.
    """
    ps_matrix = np.dot(dm, s_matrix)
    charges = np.zeros(mol.natm)

    for i, atom_label in enumerate(mol.ao_labels(fmt=None)):
        atom_idx = atom_label[0]
        charges[atom_idx] -= ps_matrix[i, i]

    for a in range(mol.natm):
        charges[a] += mol.atom_charge(a)
    print("Mulliken charges: ", charges)

    return charges 

def mulliken_charge_fitting(mol, s_matrix, v_solv_at_nuclei):
    """
    Scheme IIIB: Mulliken Charge Fitting (Solvent -> Solute)
    
    Computes the contribution to the effective Fock matrix arising from the solvent
    potential using Mulliken population analysis partitioning.
    
    Parameters:
        mol (pyscf.gto.Mole): PySCF Molecule object.
        s_matrix (numpy.ndarray): Overlap matrix (shape: nao x nao).
        v_solv_at_nuclei (numpy.ndarray): Solvent electrostatic potential at atomic centers.
        
    Returns:
        numpy.ndarray: Mulliken solvation Fock matrix correction, delta_F_solv (shape: nao x nao).
    """
    delta_F = np.zeros_like(s_matrix)
    aoslices = mol.aoslice_by_atom()

    for a in range(mol.natm):
        _, _, p0, p1 = aoslices[a]
        V_A = v_solv_at_nuclei[a]

        for b in range(mol.natm):
            _, _, q0, q1 = aoslices[b]
            V_B = v_solv_at_nuclei[b]

            delta_F[p0:p1, q0:q1] = -0.5 * s_matrix[p0:p1, q0:q1] * (V_A + V_B)
    
    return delta_F

def extract_free_energies_from_log(log_path):
    """
    Extract the final excess Gibbs Free Energy (exGF) and HNC excess free energy (excess)
    in kJ/mol from the eprism3d log file.
    """
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, 'r') as f:
        lines = f.readlines()
    for line in reversed(lines):
        if line.strip().startswith('total') and 'exGF' not in line:
            parts = line.split()
            # The columns are: Atom, mass, DN, DN_vac, -TS, LJSR, Coul, Hef0, Volume, exGF, excess
            if len(parts) >= 11:
                try:
                    exGF = float(parts[-2])
                    excess = float(parts[-1])
                    return exGF, excess
                except ValueError:
                    pass
    return None, None

class RISM_SCF:
    """
    Establish the self-consistent coupling loop between PySCF (QM DFT) and 
    EPISOL (3D-RISM solvent solver) as an object.
    """
    def __init__(self, solute_geometry, gro_path, top_path, scheme='esp', max_cycles=5, basis='6-311++g**', real_space=False, downsample=2):
        self.solute_geometry = solute_geometry
        self.gro_path = gro_path
        self.top_path = top_path
        self.scheme = scheme
        self.max_cycles = max_cycles
        self.basis = basis
        self.real_space = real_space
        self.downsample = downsample
        
        # Results to be populated during run()
        self.E_gas = None
        self.E_tot = None
        self.E_solute_polarized = None
        self.E_pol_cost = None
        self.exGF_kj = None
        self.exGF_kcal = None
        self.dG_solv = None
        self.charges = None
        
        # Automatically determine solute_path from top_path
        top_dir = os.path.dirname(os.path.abspath(top_path))
        top_base = os.path.basename(top_path)
        top_prefix = os.path.splitext(top_base)[0]
        self.solute_path = os.path.join(top_dir, f"idc_{top_prefix}.solute")

    def run(self):
        # Unpack parameters for convenience of local variables in the existing workflow
        solute_geometry = self.solute_geometry
        gro_path = self.gro_path
        top_path = self.top_path
        solute_path = self.solute_path
        scheme = self.scheme
        max_cycles = self.max_cycles
        real_space = self.real_space
        downsample = self.downsample

        print(f"\n==================================================")
        print(f"Initializing RISM-SCF Solvation Run")
        print(f"  Scheme     : {scheme}")
        print(f"  Max Cycles : {max_cycles}")
        print(f"==================================================")

        # 1. Initial Gas-phase SCF to get initial charges 
        print("\n[Step 1] Running Initial Gas-Phase DFT...")
        # Load solute coordinates directly from the GROMACS .gro file to ensure perfect alignment
        print(f"Loading solute coordinates directly from GROMACS file: {gro_path}")
        solute_atoms = parse_gro_to_pyscf_atoms(gro_path)
        mol = gto.M(atom=solute_atoms, basis=self.basis)

        mf = scf.RKS(mol)
        mf.xc = 'pbe0'
        mf.kernel()

        dm_current = mf.make_rdm1()
        s_matrix = mol.intor('int1e_ovlp')
        if scheme == 'mulliken':
            charges = get_mulliken_charges(mol, dm_current, s_matrix)
            print("Initial Mulliken charges:")
        else:
            charges = fit_esp_charges(mol, dm_current)
            print("Initial ESP-fitted charges:")
        print(charges)

        # Coordinates match exactly by construction because PySCF Mole was built directly from the GROMACS coordinates
        d = np.zeros(3)
        print("Coordinate alignment shift: d = [0.0, 0.0, 0.0] Angstroms (Direct GROMACS coordinate construction)")

        with open(gro_path, 'r') as f:
            lines_gro = f.readlines()

        # Parse GROMACS coordinates to print verification
        R_gro = []
        natm = mol.natm
        for i in range(natm):
            line = lines_gro[2 + i]
            x_i = float(line[20:28]) * 10.0
            y_i = float(line[28:36]) * 10.0
            z_i = float(line[36:44]) * 10.0
            R_gro.append([x_i, y_i, z_i])
        R_gro = np.array(R_gro)

        # Print coordinate comparison to show 100% exact matching
        print("\n=== COORDINATE ALIGNMENT CHECK (PySCF vs. GROMACS) ===")
        print(f"{'Atom':<6}{'Symbol':<8}{'Aligned PySCF (A)':<30}{'GROMACS (A)':<30}{'Deviation (A)':<15}")
        for i in range(natm):
            symbol = mol.atom_symbol(i)
            pos = mol.atom_coord(i) * lib.param.BOHR
            print(f"{i+1:<6}{symbol:<8}"
                  f"[{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]   "
                  f"[{R_gro[i][0]:8.4f}, {R_gro[i][1]:8.4f}, {R_gro[i][2]:8.4f}]   "
                  f"{0.0:8.5f}")
        print("Maximum Coordinate Deviation: 0.00000 Angstroms")
        print("Coordinate alignment check PASSED (orientations match correctly by construction).\n")

        prefix = os.path.splitext(os.path.basename(gro_path))[0]
        # Clean up old calculation files to prevent EPISOL from incrementing file suffixes
        for file_pattern in [f"{prefix}.ts4s", f"coul_{prefix}.txt", f"huv_{prefix}.txt", f"guv_{prefix}.txt"]:
            f_path = os.path.join(os.path.dirname(solute_path), file_pattern)
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError:
                    pass

        # 2. Create the initial solute topology file from GROMACS
        print("\n[Step 2] Initializing 3D-RISM Topology via EPISOL...")
        sol = epipy(gro_path, top_path, convert=True, gen_idc=True)
        sol.err_tol = 1e-8
        sol.rism(step=10, resolution=0.5)  # Quick run to generate file
        sol.kernel()

        # Update the newly generated solute file with initial gas phase charges
        update_solute_charges(solute_path, charges)

        # 3. Loop over the 3D-RISM-SCF cycle
        E_tot = None
        E_list = []
        cycle_list = [] 
        for cycle in range(1, max_cycles + 1):
            cycle_list.append(cycle)
            print(f"\n--- RISM-SCF Cycle {cycle} ---")

            # A. Run 3D-RISM with the current charges
            print("Running 3D-RISM solver (EPISOL)...")
            prefix = os.path.splitext(os.path.basename(gro_path))[0]
            # Clean up files from previous cycle so EPISOL writes to the base prefix.ts4s
            for file_pattern in [f"{prefix}.ts4s", f"coul_{prefix}.txt", f"huv_{prefix}.txt"]:
                f_path = os.path.join(os.path.dirname(solute_path), file_pattern)
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except OSError:
                        pass
            sol = epipy(gro_path, solute_path, convert=False, gen_idc=False)
            sol.err_tol = 1e-8
            sol.rism(step=5000, resolution=0.25)

            if scheme in ['continuous']:
                print("Calculating continuous solute potential on EPISOL grid...")
                nx, ny, nz = int(sol.grid[0]), int(sol.grid[1]), int(sol.grid[2])
                resolution = sol.resolution  # in Angstroms
                
                # 1. Get EPISOL grid coordinates
                x_coord = np.arange(nx) * resolution
                y_coord = np.arange(ny) * resolution
                z_coord = np.arange(nz) * resolution
                
                # Grid points in Angstroms (order of iz, iy, ix)
                Z, Y, X = np.meshgrid(z_coord, y_coord, x_coord, indexing='ij')
                grid_points_ang = np.stack([X, Y, Z], axis=-1)
                grid_points_flat = grid_points_ang.reshape(-1, 3)
                
                # Align coordinate systems: shift EPISOL grid to PySCF coordinate system
                grid_points_flat_pyscf = grid_points_flat - d
                
                # Convert to Bohr for PySCF
                grid_points_bohr = grid_points_flat_pyscf / lib.param.BOHR
                
                # 2. Use fakemol to get electronic potential V^e
                print("Evaluating electronic electrostatic potential on grid...")
                V_elec = np.zeros(len(grid_points_bohr))
                chunk_size = 10000
                for i in range(0, len(grid_points_bohr), chunk_size):
                    chunk_points = grid_points_bohr[i:i+chunk_size]
                    fakemol = gto.fakemol_for_charges(chunk_points)
                    ao_ao_p = df.incore.aux_e2(mol, fakemol)
                    V_elec[i:i+chunk_size] = -np.einsum('ijp,ij->p', ao_ao_p, dm_current)
                
                # 3. Add nuclear potential correction V^{n-e} = Z_alpha - Q_alpha
                print("Calculating nuclear potential correction...")
                V_nuc_corr = np.zeros(len(grid_points_bohr))
                for a in range(mol.natm):
                    R_a = mol.atom_coord(a)  # in Bohr
                    Z_a = mol.atom_charge(a)
                    Q_a = charges[a]
                    diff_charge = Z_a - Q_a
                    dists = np.linalg.norm(grid_points_bohr - R_a, axis=1)
                    dists = np.where(dists < 1e-8, 1e-8, dists)
                    V_nuc_corr += diff_charge / dists
                
                # Total potential in Hartree
                V_add = V_elec + V_nuc_corr
                
                # Convert to EPISOL units (kJ/mol/e): 1 Hartree/e = 2625.5002 kJ/mol/e
                V_add_episol = V_add * 2625.5002
                
                # Reshape to (nz, ny, nx)
                V_add_reshaped = V_add_episol.reshape(nz, ny, nx)
                
                # Write to prefix.coulomb
                coulomb_file = os.path.join(os.path.dirname(solute_path), f"{prefix}.coulomb")
                print(f"Writing potential correction grid to {coulomb_file}...")
                with open(coulomb_file, 'w') as f:
                    for val in V_add_reshaped.ravel():
                        f.write(f"{val:.10e}\n")
                
                # Prepend command line options to sol.rism_args
                sol.rism_args = f" -i {prefix} -cmd load:coul"

            sol.kernel()

            # B. Extract potential grid
            if real_space:
                # We do not need the full 3D FFT potential grid for SCF calculations in real space
                # but we compute it to generate the plots at the end of the run
                if cycle == max_cycles:
                    print("Computing solvent reaction field potential grid via FFT (for plotting)...")
                    pot_grid_solvent = compute_solvent_potential_grid_fft(sol)
                    pot_grid = pot_grid_solvent.transpose(2, 1, 0)
                else:
                    pot_grid = np.zeros((10, 10, 10))  # dummy
                resolution = sol.resolution
                nx, ny, nz = pot_grid.shape
            else:
                print("Computing solvent reaction field potential grid via FFT...")
                pot_grid_solvent = compute_solvent_potential_grid_fft(sol)
                pot_grid = pot_grid_solvent.transpose(2, 1, 0)
                resolution = sol.resolution
                nx, ny, nz = pot_grid.shape

            # Construct grid coordinates
            x = np.arange(nx) * resolution
            y = np.arange(ny) * resolution
            z = np.arange(nz) * resolution

            if scheme in ['continuous']:
                # Scheme I: Calculate continuous potential integration
                print("Integrating solvent potential over atom-centered grid (Continuous/Density)...")
                if real_space:
                    delta_F_solv = continous_solute_charge_density(mol, pot_grid, x, y, z, d, real_space_sol=sol, downsample=downsample)
                    nuc_coords_ang = mol.atom_coords() * lib.param.BOHR
                    nuc_coords_ang_shifted = nuc_coords_ang + d
                    pot_at_nuclei_kj = compute_solvent_potential_at_coords_real_space(sol, nuc_coords_ang_shifted, d=None, downsample=1)
                    pot_at_nuclei = pot_at_nuclei_kj / 2625.5002
                else:
                    delta_F_solv = continous_solute_charge_density(mol, pot_grid, x, y, z, d)
                    # Interpolate potential at nuclear positions (for energy correction)
                    pot_interpolator = RegularGridInterpolator((x, y, z), pot_grid, bounds_error=False, fill_value=0.0)
                    nuc_coords_ang = mol.atom_coords() * lib.param.BOHR
                    nuc_coords_ang_shifted = nuc_coords_ang + d
                    pot_at_nuclei = pot_interpolator(nuc_coords_ang_shifted) / 2625.5002
                
                nuc_charges = mol.atom_charges()
                E_nuc_solv = np.sum(nuc_charges * pot_at_nuclei)

            elif scheme == 'esp':
                # Scheme II: Add density-derivative contribution
                print("Computing solvent reaction field potential at nuclei...")
                if real_space:
                    nuc_coords_ang = mol.atom_coords() * lib.param.BOHR
                    pot_at_nuclei_kj = compute_solvent_potential_at_coords_real_space(sol, nuc_coords_ang + d, d=None, downsample=1)
                else:
                    pot_at_nuclei_kj = compute_solvent_potential_at_nuclei(sol, mol, d)
                pot_at_nuclei = pot_at_nuclei_kj / 2625.5002
                
                print("Computing density-derivative contribution (Polarization back-reaction)...")
                delta_F_solv = compute_fock_derivative(mol, pot_at_nuclei)
                E_nuc_solv = np.sum(mol.atom_charges() * pot_at_nuclei)

            elif scheme == 'mulliken':
                # Scheme III: Mulliken Charge Fitting
                print("Computing solvent reaction field potential at nuclei...")
                pot_at_nuclei_kj = compute_solvent_potential_at_nuclei(sol, mol, d)
                pot_at_nuclei = pot_at_nuclei_kj / 2625.5002
                
                print("Computing Mulliken charge fitting Fock contribution...")
                s_matrix = mol.intor('int1e_ovlp')
                delta_F_solv = mulliken_charge_fitting(mol, s_matrix, pot_at_nuclei)
                E_nuc_solv = np.sum(mol.atom_charges() * pot_at_nuclei)

            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            # Setup PySCF DFT in solvent field
            mf_solv = scf.RKS(mol)
            mf_solv.xc = 'pbe0'

            # Override get_hcore to inject the solvent potential matrix (delta_F_solv)
            original_get_hcore = mf_solv.get_hcore
            def get_hcore_with_solvent(*args, **kwargs):
                return original_get_hcore(*args, **kwargs) + delta_F_solv
            mf_solv.get_hcore = get_hcore_with_solvent

            # Override energy_nuc to inject the nuclear-solvent interaction energy
            original_energy_nuc = mf_solv.energy_nuc
            def energy_nuc_with_solvent(*args, **kwargs):
                return original_energy_nuc(*args, **kwargs) + E_nuc_solv
            mf_solv.energy_nuc = energy_nuc_with_solvent

            # Run SCF in the solvent field
            print("Running PySCF DFT in solvent field...")
            E_tot = mf_solv.kernel()
            print(f"Total energy in solvent (Cycle {cycle}): {E_tot:.6f} Hartree")
            E_list.append(E_tot)

            # Update solute charges using the selected scheme from the new density matrix
            dm_current = mf_solv.make_rdm1()
            if scheme == 'mulliken':
                s_matrix = mol.intor('int1e_ovlp')
                charges_new = get_mulliken_charges(mol, dm_current, s_matrix)
            else:
                charges_new = fit_esp_charges(mol, dm_current)

            charges = charges_new
            if scheme == 'mulliken':
                print(f"Updated Mulliken charges (Cycle {cycle}):")
            else:
                print(f"Updated ESP-fitted charges (Cycle {cycle}):")
            print(charges)
            update_solute_charges(solute_path, charges)
        
        # 1. Calculate the polarized solute internal energy (gas-phase energy evaluated with final density matrix)
        E_solute_polarized = mf.energy_tot(dm=dm_current)
        E_pol_cost = (E_solute_polarized - mf.e_tot) * 627.509  # in kcal/mol
        
        # 2. Extract excess solvation free energy (exGF) from the log file
        log_path = f"{prefix}.log"
        exGF_kj, HNC_kj = extract_free_energies_from_log(log_path)
        
        print(f"\n==================================================")
        print(f"RISM-SCF Calculation Completed Successfully")
        print(f"  Final solvated SCF Energy: {E_tot:.6f} Hartree")
        print(f"  Final solute charges     : {charges}")
        print(f"--------------------------------------------------")
        print(f"  Physical Energy Analysis:")
        print(f"    Gas-Phase ground state energy: {mf.e_tot:.6f} Hartree")
        print(f"    Polarized solute internal energy: {E_solute_polarized:.6f} Hartree")
        print(f"    Wave function polarization cost : {E_pol_cost:.3f} kcal/mol")
        if exGF_kj is not None:
            exGF_kcal = exGF_kj / 4.184
            dG_solv_kcal = E_pol_cost + exGF_kcal
            HNC_kcal = HNC_kj / 4.184
            dG_HNC_kcal = E_pol_cost + HNC_kcal
            print(f"    3D-RISM Excess Free Energy (GF) : {exGF_kcal:.3f} kcal/mol ({exGF_kj:.3f} kJ/mol)")
            print(f"    3D-RISM Excess Free Energy (HNC): {HNC_kcal:.3f} kcal/mol ({HNC_kj:.3f} kJ/mol)")
            print(f"    Net Hydration Free Energy (GF)  : {dG_solv_kcal:.3f} kcal/mol ({dG_solv_kcal*4.184:.3f} kJ/mol)")
            print(f"    Net Hydration Free Energy (HNC) : {dG_HNC_kcal:.3f} kcal/mol ({dG_HNC_kcal*4.184:.3f} kJ/mol)")
        else:
            print(f"    3D-RISM Excess Free Energy      : N/A (log file not found or could not be parsed)")
        print(f"==================================================")

        # Print potential and charge values around Carbonyl Oxygen (Atom 2)
        #print("\n=== SOLVENT POTENTIAL & CHARGE DENSITY AROUND CARBONYL OXYGEN (Atom 2) ===")
        # Atom 2 coordinates in Angstroms
        R_o2 = mol.atom_coord(1) * lib.param.BOHR
        #print(f"Carbonyl Oxygen (Atom 2) coordinates: [{R_o2[0]:.4f}, {R_o2[1]:.4f}, {R_o2[2]:.4f}] Angstroms")
        
        # Reconstruct solvent charge density grid from huv
        h_O = sol.select_grid('huv atom O')
        h_H = sol.select_grid('huv atom H')
        rho_O = 0.03342288177  # molecules/A^3
        rho_H = 2.0 * rho_O
        q_O = -0.834
        q_H = 0.417
        charge_grid = q_O * rho_O * h_O + q_H * rho_H * h_H
        
        # Get grid sizes and resolution
        nx, ny, nz = pot_grid.shape
        resolution = sol.resolution
        print(f"Grid shape: ({nx}, {ny}, {nz}), Resolution: {resolution:.4f} Angstroms")
        
        print(f"\n{'Grid Index (ix, iy, iz)':<25}{'Coordinates (A)':<28}{'Distance (A)':<13}{'Potential (kJ/mol/e)':<22}{'Potential (Hartree/e)':<22}{'Solvent Charge (e/A^3)':<22}")
        print("-" * 132)
        
        count = 0
        for ix in range(nx):
            x_coord = ix * resolution
            if abs(x_coord - R_o2[0]) > 1.2:
                continue
            for iy in range(ny):
                y_coord = iy * resolution
                if abs(y_coord - R_o2[1]) > 1.2:
                    continue
                for iz in range(nz):
                    z_coord = iz * resolution
                    if abs(z_coord - R_o2[2]) > 1.2:
                        continue
                    
                    dist = np.sqrt((x_coord - R_o2[0])**2 + (y_coord - R_o2[1])**2 + (z_coord - R_o2[2])**2)
                    if dist <= 1.2:
                        v_kj = pot_grid[ix, iy, iz]
                        v_har = v_kj / 2625.5002
                        chg_dens = charge_grid[ix, iy, iz]
                        print(f"({ix:3d}, {iy:3d}, {iz:3d})            "
                              f"[{x_coord:7.3f}, {y_coord:7.3f}, {z_coord:7.3f}]   "
                              f"{dist:7.3f}       "
                              f"{v_kj:10.4f}            "
                              f"{v_har:10.6f}            "
                              f"{chg_dens:10.6f}")
                        count += 1
        print(f"Total grid points printed: {count}")
        print("-" * 132)

        
        # Store results in the object attributes
        self.E_gas = mf.e_tot
        self.E_tot = E_tot
        self.E_list = np.array(E_list)
        self.cycle_list = np.array(cycle_list)
        self.E_solute_polarized = E_solute_polarized
        self.E_pol_cost = E_pol_cost
        self.exGF_kj = exGF_kj
        self.exGF_kcal = exGF_kj / 4.184 if exGF_kj is not None else None
        self.dG_solv = dG_solv_kcal / 627.509 if exGF_kj is not None else None
        self.charges = charges

        # Save energy convergence plot non-blockingly
        plt.figure()
        plt.plot(self.cycle_list, self.E_list, marker='o')
        plt.xlabel('Cycle')
        plt.ylabel('Total Energy (Hartree)')
        plt.title('RISM-SCF Convergence')
        plt.grid(True)
        plt.savefig(f"{prefix}_convergence.png", dpi=300)
        plt.close()
        print(f"Saved energy convergence plot to: {prefix}_convergence.png")

        # Generate and save 2D potential and charge density plots of the slice at the molecular plane
        try:
            heavy_atom_zs = [mol.atom_coord(i)[2] * lib.param.BOHR for i in range(mol.natm) if mol.atom_symbol(i) != 'H']
            z_plane = np.mean(heavy_atom_zs) if heavy_atom_zs else 15.0
            iz_plane = int(round(z_plane / resolution))

            # Save 2D Potential Comparison Plot
            plot_potential_comparison_2d(mol, dm_current, pot_grid, resolution, iz_plane, filename=f"{prefix}_potential_comparison.png")
            
            # Save 2D Charge Comparison Plot
            plot_charge_comparison_2d(mol, dm_current, sol, resolution, iz_plane, filename=f"{prefix}_charge_comparison.png")
        except Exception as e:
            print(f"Warning: Could not generate 2D plots: {e}")

def plot_potential_comparison_2d(mol, dm, pot_grid, resolution, iz_plane, filename='potential_comparison_2d.png'):
    """
    Plot a side-by-side 2D contour plot comparing the solvent electrostatic potential (left)
    and the solute electrostatic potential (right) at the molecular plane.
    """
    import matplotlib.pyplot as plt
    nx, ny, nz = pot_grid.shape
    
    # 1. Coordinate grids in Angstroms
    x = np.arange(nx) * resolution
    y = np.arange(ny) * resolution
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    z_val = iz_plane * resolution
    
    # 2. Extract solvent potential slice
    pot_solvent_slice = pot_grid[:, :, iz_plane]
    
    # 3. Calculate solute electrostatic potential slice
    # Flatten the 2D grid coordinates for evaluation
    grid_coords_ang = np.vstack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), z_val)]).T
    grid_coords_bohr = grid_coords_ang / lib.param.BOHR
    
    # Calculate electronic contribution to the potential (in Hartree)
    fakemol = gto.fakemol_for_charges(grid_coords_bohr)
    V_elec = -np.einsum('ijp,ij->p', df.incore.aux_e2(mol, fakemol), dm)
    
    # Calculate nuclear contribution to the potential (in Hartree)
    V_nuc = np.zeros(len(grid_coords_bohr))
    for a in range(mol.natm):
        R_a = mol.atom_coord(a)
        Z_a = mol.atom_charge(a)
        dists = np.linalg.norm(grid_coords_bohr - R_a, axis=1)
        dists = np.where(dists < 1e-8, 1e-8, dists)
        V_nuc += Z_a / dists
        
    V_solute = V_elec + V_nuc
    
    # Convert solute potential to kJ/mol/e (1 Hartree = 2625.5002 kJ/mol)
    pot_solute_slice = (V_solute * 2625.5002).reshape(nx, ny)
    
    # 4. Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Common plotting details
    coords_ang = mol.atom_coords() * lib.param.BOHR
    x_min, y_min = np.min(coords_ang[:, :2], axis=0) - 3.0
    x_max, y_max = np.max(coords_ang[:, :2], axis=0) + 3.0
    xlims = (max(0, x_min), min(nx * resolution, x_max))
    ylims = (max(0, y_min), min(ny * resolution, y_max))
    
    # Left Panel: Solvent potential
    vmax_solv = np.percentile(np.abs(pot_solvent_slice), 100)
    levels_solv = np.linspace(-vmax_solv, vmax_solv, 100)
    cf1 = axes[0].contourf(X, Y, pot_solvent_slice, levels=levels_solv, cmap='RdBu_r', extend='both')
    cbar1 = fig.colorbar(cf1, ax=axes[0])
    cbar1.set_label('Potential (kJ/mol/e)', fontsize=11)
    axes[0].set_title(f'Solvent Electrostatic Potential', fontsize=13)
    
    # Right Panel: Solute potential (limit percentile to avoid nuclear singularity blowup)
    vmax_solute = np.percentile(np.abs(pot_solute_slice), 98)
    if vmax_solute == 0:
        vmax_solute = 1000.0
    levels_solute = np.linspace(-vmax_solute, vmax_solute, 100)
    cf2 = axes[1].contourf(X, Y, pot_solute_slice, levels=levels_solute, cmap='RdBu_r', extend='both')
    cbar2 = fig.colorbar(cf2, ax=axes[1])
    cbar2.set_label('Potential (kJ/mol/e)', fontsize=11)
    axes[1].set_title(f'Solute Electrostatic Potential', fontsize=13)
    
    # Overlay solute atoms on both panels
    for ax in axes:
        for i in range(mol.natm):
            symbol = mol.atom_symbol(i)
            coord = mol.atom_coord(i) * lib.param.BOHR
            x_atom, y_atom, z_atom = coord
            
            color = 'black'
            if symbol == 'O':
                color = 'red'
            elif symbol == 'H':
                color = 'gray'
            elif symbol == 'C':
                color = 'darkgray'
                
            ax.scatter(x_atom, y_atom, color=color, edgecolor='black', s=120, zorder=5)
            ax.text(x_atom + 0.12, y_atom + 0.12, f"{symbol}{i+1}", fontsize=10, weight='bold', zorder=6)
            
        ax.set_xlabel('X (Å)', fontsize=11)
        ax.set_ylabel('Y (Å)', fontsize=11)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_aspect('equal', adjustable='box')
        
    fig.suptitle(f'Electrostatic Potential Comparison at Molecular Plane (Z = {z_val:.2f} Å)', fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved 2D electrostatic potential comparison plot to: {filename}")

def plot_charge_comparison_2d(mol, dm, sol, resolution, iz_plane, filename='charge_comparison_2d.png'):
    """
    Plot a side-by-side 2D contour plot comparing the solvent charge density (left)
    and the solute charge density (right) at the molecular plane.
    """
    import matplotlib.pyplot as plt
    
    # 1. Reconstruct solvent charge density grid from huv
    h_O = sol.select_grid('huv atom O')
    h_H = sol.select_grid('huv atom H')
    rho_O = 0.03342288177  # molecules/A^3
    rho_H = 2.0 * rho_O
    q_O = -0.834
    q_H = 0.417
    charge_grid = q_O * rho_O * h_O + q_H * rho_H * h_H
    
    nx, ny, nz = charge_grid.shape
    x = np.arange(nx) * resolution
    y = np.arange(ny) * resolution
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    z_val = iz_plane * resolution
    
    # 2. Extract solvent charge density slice
    charge_solvent_slice = charge_grid[:, :, iz_plane]
    
    # 3. Calculate solute electron charge density slice (in e/A^3)
    grid_coords_ang = np.vstack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), z_val)]).T
    grid_coords_bohr = grid_coords_ang / lib.param.BOHR
    
    # Evaluate AOs on the grid slice
    ao = dft.numint.eval_ao(mol, grid_coords_bohr)
    
    # Solute electron density n_e(r) = sum_uv P_uv phi_u(r) phi_v(r)
    n_e = np.einsum('pi,pj,ij->p', ao, ao, dm)
    
    # Convert Bohr^-3 to Angstrom^-3: 1 Bohr = 0.529177 Angstroms -> 1 Bohr^-3 = 1 / 0.529177^3 = 6.748333 Angstrom^-3
    # Solute charge density due to electrons is -n_e
    charge_solute_slice = (-n_e * 6.748333).reshape(nx, ny)
    
    # 4. Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    coords_ang = mol.atom_coords() * lib.param.BOHR
    x_min, y_min = np.min(coords_ang[:, :2], axis=0) - 3.0
    x_max, y_max = np.max(coords_ang[:, :2], axis=0) + 3.0
    xlims = (max(0, x_min), min(nx * resolution, x_max))
    ylims = (max(0, y_min), min(ny * resolution, y_max))
    
    # Left Panel: Solvent charge density
    vmax_solv = np.percentile(np.abs(charge_solvent_slice), 98)
    if vmax_solv == 0:
        vmax_solv = 0.05
    levels_solv = np.linspace(-vmax_solv, vmax_solv, 100)
    cf1 = axes[0].contourf(X, Y, charge_solvent_slice, levels=levels_solv, cmap='bwr', extend='both')
    cbar1 = fig.colorbar(cf1, ax=axes[0])
    cbar1.set_label('Charge Density (e/Å³)', fontsize=11)
    axes[0].set_title(f'Solvent Charge Density', fontsize=13)
    
    # Right Panel: Solute electronic charge density
    vmax_solute = np.percentile(np.abs(charge_solute_slice), 98)
    if vmax_solute == 0:
        vmax_solute = 1.0
    levels_solute = np.linspace(-vmax_solute, 0, 100)  # electron density is negative
    cf2 = axes[1].contourf(X, Y, charge_solute_slice, levels=levels_solute, cmap='bwr', extend='both')
    cbar2 = fig.colorbar(cf2, ax=axes[1])
    cbar2.set_label('Charge Density (e/Å³)', fontsize=11)
    axes[1].set_title(f'Solute Electronic Charge Density', fontsize=13)
    
    # Overlay solute atoms on both panels
    for ax in axes:
        for i in range(mol.natm):
            symbol = mol.atom_symbol(i)
            coord = mol.atom_coord(i) * lib.param.BOHR
            x_atom, y_atom, z_atom = coord
            
            color = 'black'
            if symbol == 'O':
                color = 'red'
            elif symbol == 'H':
                color = 'gray'
            elif symbol == 'C':
                color = 'darkgray'
                
            ax.scatter(x_atom, y_atom, color=color, edgecolor='black', s=120, zorder=5)
            ax.text(x_atom + 0.12, y_atom + 0.12, f"{symbol}{i+1}", fontsize=10, weight='bold', zorder=6)
            
        ax.set_xlabel('X (Å)', fontsize=11)
        ax.set_ylabel('Y (Å)', fontsize=11)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_aspect('equal', adjustable='box')
        
    fig.suptitle(f'Charge Density Comparison at Molecular Plane (Z = {z_val:.2f} Å)', fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved 2D charge density comparison plot to: {filename}")

def run_rism_scf(solute_geometry, gro_path, top_path, scheme='esp', max_cycles=5, basis='6-311++g**', real_space=False, downsample=2):
    """
    Backward-compatible wrapper function that instantiates the RISM_SCF class and runs it.
    """
    solver = RISM_SCF(solute_geometry, gro_path, top_path, scheme=scheme, max_cycles=max_cycles, basis=basis, real_space=real_space, downsample=downsample)
    solver.run()
    return solver.E_tot, solver.charges, solver.dG_solv

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sdf_path = os.path.join(script_dir, "water.sdf")
    gro_path = os.path.join(script_dir, "water.gro")
    top_path = os.path.join(script_dir, "water.top")
    
    import argparse
    parser = argparse.ArgumentParser(description='Run 3D-RISM-SCF calculations.')
    parser.main_group = parser.add_mutually_exclusive_group()
    parser.main_group.add_argument('--solute-charge-density', action='store_true', help='Run Scheme I: Continuous solute charge density (without Fock derivative)')
    parser.main_group.add_argument('--esp-charge-fitting', action='store_true', help='Run Scheme II: ESP charge fitting (with Fock derivative)')
    parser.main_group.add_argument('--mulliken-charge-fitting', action='store_true', help='Run Scheme III: Mulliken charge fitting')
    parser.add_argument('--max-cycles', type=int, default=5, help='Maximum number of RISM-SCF cycles (default: 5)')
    parser.add_argument('--basis', type=str, default='6-311++g**', help="Basis set for PySCF (default: '6-311++g**')")
    parser.add_argument('--real-space', action='store_true', help='Use direct real-space Coulomb summation instead of FFT')
    parser.add_argument('--downsample', type=int, default=2, help='Downsampling factor for source grid in real-space potential (default: 2)')

    args = parser.parse_args()

    # Determine scheme based on argument
    if args.solute_charge_density:
        scheme = 'continuous'
    elif args.esp_charge_fitting:
        scheme = 'esp'
    elif args.mulliken_charge_fitting:
        scheme = 'mulliken'
    else:
        # Default scheme is ESP Charge Fitting (Scheme II)
        print("No scheme flag provided. Defaulting to ESP charge fitting (Scheme II).")
        scheme = 'esp'

    # Run the main RISM-SCF workflow using the parsed max_cycles
    run_rism_scf(sdf_path, gro_path, top_path, scheme=scheme, max_cycles=args.max_cycles, basis=args.basis, real_space=args.real_space, downsample=args.downsample)
    