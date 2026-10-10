import os
import sys
import numpy as np
from pyscf import gto, lib

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

from episol import epipy
from rism_scf import compute_solvent_potential_at_nuclei

from rism_scf import parse_gro_to_pyscf_atoms

# 1. Load PySCF molecule directly from GROMACS coordinates
atoms = parse_gro_to_pyscf_atoms("water.gro")
mol = gto.M(atom=atoms, basis='6-311g')

# 2. Coordinate alignment shift (exactly zero by construction)
d = np.zeros(3)

# 3. Load EPISOL state (generate idc_water.solute from water.top if missing)
if not os.path.exists("idc_water.solute"):
    sol = epipy("water.gro", "water.top", convert=True, gen_idc=True)
else:
    sol = epipy("water.gro", "idc_water.solute", convert=False, gen_idc=False)
sol.rism(step=10, resolution=0.25)
sol.kernel()

# 4. Compute potentials at nuclei (in kJ/mol/e)
pot_at_nuclei_kj = compute_solvent_potential_at_nuclei(sol, mol, d)

# 5. Print results
print("\n=== SOLVENT POTENTIAL AT SOLUTE NUCLEI (kJ/mol/e) ===")
for a in range(mol.natm):
    symbol = mol.atom_symbol(a)
    coord = mol.atom_coord(a) * lib.param.BOHR
    print(f"Atom {a+1:d} ({symbol:s}) at coord {coord}: {pot_at_nuclei_kj[a]:.4f} kJ/mol/e")