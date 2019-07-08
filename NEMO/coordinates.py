import numpy as np
import string

# Alphabets
alphabet = 'ACDEFGHIKLMNPQRSTVWY'
AAcodes_invertible = {  # Standard 20
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
           'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
           'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
           # 21st and 22nd amino acids
           'CSE': 'U',  # Selenocysteine
           'PYL': 'O',  # Pyrrolysine
           # Ambiguity codes
           'GLX': 'Z',  # Glutamine or Glutamate
           'ASX': 'B',  # Asparagine or Aspartate
           'XLE': 'J',  # Leucine or Isoleucine
           # Codes where the heavy atoms change
           'MSE': 'm',  # Selenomethionine
           'HYP': 'p'  # Hydroxyproline
}
AAcodes_inverse = {val: key for key, val in AAcodes_invertible.iteritems()}

def alignment_rotation_reflection(X_input, X_target):
    # Orthogonal Procrustes (ie Kabsch w reflection). Allows reflection by not enforcing det > 0.
    X_target_centered = X_target - np.mean(X_target, axis=1, keepdims=True)
    X_centered = X_input - np.mean(X_input, axis=1, keepdims=True)
    A = np.dot(X_centered, X_target_centered.T)
    U, _, V = np.linalg.svd(A, full_matrices=True)
    # While this would normally just be V, np.linalg.svd returns V.T not V
    R = np.dot(V.T, U.T)
    return R

def alignment_rotation(X_input, X_target):
    # Kabsch algorithm
    X_target_centered = X_target - np.mean(X_target, axis=1, keepdims=True)
    X_centered = X_input - np.mean(X_input, axis=1, keepdims=True)
    A = np.dot(X_centered, X_target_centered.T)
    U, _, V = np.linalg.svd(A, full_matrices=True)
    # While this would normally just be V, np.linalg.svd returns V.T not V
    D = np.diag([1, 1, np.linalg.det(np.dot(V.T, U.T))])
    # Enforce non-reflection
    R = np.dot(np.dot(V.T, D), U.T)
    return R

def extract_coords_sets(X):
    """Extact five (3,L) coordinate sets from flattened 5-bead model (L,15)"""
    L = X.shape[0]
    X_unflat = X.reshape((L,5,3))
    X_sets = np.transpose(X_unflat, [1,2,0])
    coords_sets = [X_sets[i] for i in range(5)]
    return coords_sets

def batch_trajectory_to_pdb(lengths, sequences, X_target, X_trajectory, masks, skip=2,
    grid_space_x = 120., grid_space_y = 60., grid_skip = 50., draw_target=True):
    """ Write a coarse-grained folding trajectory to a PDB file """

    # Preparations
    sequence_list = []
    rotation_list = []
    batch_size = X_target.shape[0]

    # Output a maximum of 16 chains
    max_count = 16
    if batch_size > max_count:
        lengths = lengths[0:max_count]
        sequences = sequences[0:max_count]
        X_target = X_target[0:max_count]
        X_trajectory = X_trajectory[0:max_count]
        masks = masks[0:16]
        batch_size = max_count

    # Remove any non-AA alphabet character
    sequences = sequences[:,:,:20]

    for i in xrange(batch_size):
        # One-Hot encoding back to sequence
        L = lengths[i]
        sequence_onehot = sequences[i,0:L,:]
        sequence_bits = sequence_onehot.tolist()
        sequence = [alphabet[bits.index(1)] if 1 in bits else ' ' for bits in sequence_bits]
        sequence = ''.join(sequence)
        sequence_list.append(sequence)

        # # Compute optimal rotation (& potentially reflection) matrix
        beads_target = X_target[i,:,:]
        beads_model = X_trajectory[i,-1,:,:]
        # Mask it
        mask = masks[i,:]
        beads_target = beads_target[:, mask > 0]
        beads_model = beads_model[:, mask > 0]
        if draw_target:
            R = alignment_rotation(beads_model, beads_target)
            rotation_list.append(R)

    num_columns = np.round(np.sqrt(batch_size))
    num_rows = np.round(batch_size / num_columns)
    num_frames = X_trajectory.shape[1]

    # Assemble the pdb file
    pdb_lines = []
    chain_names = list(string.ascii_uppercase + string.digits)
    for t in xrange(0, num_frames, skip):
        # PDB can handle at most 9999 frames
        assert(t < 10000)
        pdb_lines += ['MODEL     ' + str(t+1).rjust(4)]
        chain_index = 0
        counter = 1
        for i in xrange(batch_size):
            L = lengths[i]
            sequence_onehot = sequences[i,0:L,:]
            mask = masks[i,0:L]
            sequence = sequence_list[i]

            # Get coordinates for the frame
            beads_target = np.copy(X_target[i,:,0:L])
            beads_model = np.copy(X_trajectory[i,t,:,0:L])

            # Center and rotate
            beads_target -= np.mean(beads_target[:,mask > 0], axis=1, keepdims=True)
            beads_model -= np.mean(beads_model[:,mask > 0], axis=1, keepdims=True)
            if draw_target:
                R = rotation_list[i]
                beads_model = np.dot(R, beads_model)

            # Grid offsets
            col_ix = i % num_columns
            row_ix = np.floor(i / num_columns)
            gridX = grid_space_x * (col_ix - num_columns / 2.)
            gridY = grid_space_y * (row_ix - num_rows / 2.)
            beads_target += np.expand_dims(np.array([gridX, gridY, 0]), axis=1)
            beads_model += np.expand_dims(np.array([gridX + grid_skip, gridY, 0]), axis=1)

            if draw_target:
                pdb_target, counter = coarse_to_pdb_sequence(beads_target, sequence, mask, chain_names[chain_index], counter)
            pdb_model, counter = coarse_to_pdb_sequence(beads_model, sequence, None, chain_names[chain_index + 1], counter)

            # Append everything
            if draw_target:
                pdb_lines = pdb_lines + pdb_target + pdb_model
                chain_index += 2
            else:
                pdb_lines = pdb_lines + pdb_model
                chain_index += 1
        # End the frame
        pdb_lines += ['ENDMDL']

    # Write a single chain
    return pdb_lines


def batch_to_pdb(lengths, sequences, coords_target, coords_model, masks,
    grid_space_x = 120., grid_space_y = 60., grid_skip = 50., draw_target=True,
    max_count = 16):
    """ Write a chain to a PDB file """
    batch_size = lengths.shape[0]

    # Output a maximum of MAX_COUNT chains
    if batch_size > max_count:
        lengths = lengths[0:max_count]
        sequences = sequences[0:max_count]
        coords_target = coords_target[0:max_count]
        coords_model = coords_model[0:max_count]
        masks = masks[0:max_count]
        batch_size = max_count

    # Remove any non-AA alphabet character
    sequences = sequences[:,:,:20]

    chain_names = list(string.ascii_uppercase + string.digits)
    chain_index = 0
    pdb_lines = []
    # Grid spacing
    num_columns = np.round(np.sqrt(batch_size))
    num_rows = np.round(batch_size / num_columns)
    for i in xrange(batch_size):
        L = lengths[i]
        X_target = coords_target[i,0:L,:]
        X_model = coords_model[i,0:L,:]
        sequence_onehot = sequences[i,0:L,:]
        mask = masks[i,0:L]

        # One-Hot encoding back to sequence
        sequence_bits = sequence_onehot.tolist()
        sequence = [alphabet[bits.index(1)] if 1 in bits else ' ' for bits in sequence_bits]
        sequence = ''.join(sequence)

        # Extract bead representations (N,CA,C,O,SC)
        beads_target = extract_coords_sets(X_target)
        beads_model = extract_coords_sets(X_model)

        # Center of mass of second bead (C-Alpha)
        center_target = np.mean(beads_target[1][:,mask > 0], axis=1, keepdims=True)
        center_model = np.mean(beads_model[1][:,mask > 0], axis=1, keepdims=True)
        for j in range(5):
            beads_target[j] -= center_target
            beads_model[j] -= center_model

        # Compute optimal rotation (& potentially reflection) matrix
        # ca_target = beads_target[1][:,mask > 0]
        # ca_model = beads_model[1][:,mask > 0]
        # R = alignment_rotation(ca_model, ca_target)
        # for j in range(5):
        #     beads_model[j] = np.dot(R, beads_model[j])

        # Grid offsets
        col_ix = i % num_columns
        row_ix = np.floor(i / num_columns)
        gridX = grid_space_x * (col_ix - num_columns / 2.)
        gridY = grid_space_y * (row_ix - num_rows / 2.)
        offset_target = np.expand_dims(np.array([gridX, gridY, 0]), axis=1)
        offset_model = np.expand_dims(np.array([gridX + grid_skip, gridY, 0]), axis=1)
        for j in range(5):
            beads_target[j] += offset_target
            beads_model[j] += offset_model
        if draw_target:
            pdb_target = coords_to_pdb_sequence(beads_target, sequence, chain_names[chain_index], mask)
        pdb_model = coords_to_pdb_sequence(beads_model, sequence, chain_names[chain_index + 1])

        # Append everything
        if draw_target:
            pdb_lines = pdb_lines + pdb_target + pdb_model
            chain_index += 2
        else:
            pdb_lines = pdb_lines + pdb_model
            chain_index += 1
        
    # Write a single chain
    return pdb_lines

def single_to_pdb(sequence, coords, mask=None):
    """ Write a chain to a PDB file """

    # Remove any non-AA alphabet character
    sequence = sequence[:,:20]
    sequence_bits = sequence.tolist()
    sequence = [alphabet[bits.index(1)] if 1 in bits else ' ' for bits in sequence_bits]
    sequence = ''.join(sequence)

    if mask is None:
        mask = np.ones((len(sequence)))

    # Extract bead representations (N,CA,C,O,SC)
    beads_model = extract_coords_sets(coords)
    pdb_model = coords_to_pdb_sequence(beads_model, sequence, 'A')
        
    # Write a single chain
    return pdb_model

def coarse_to_pdb_sequence(X, sequence, mask, chain, counter):
    """ Write a chain to a PDB file """
    pdb_lines = ''
    atom_type = ' CA '
    element = 'C'
    pdb_lines = []
    for i in xrange(X.shape[1]):
        # Determine atom code
        letter = sequence[i]
        residue_name = AAcodes_inverse[letter] if letter in AAcodes_inverse else 'CSE'
        if mask is None or mask[i]:
            # N CA C O SCOM
            #  1 -  6        Record name   "ATOM  "
            line = 'ATOM  '
            # 7 - 11        Integer       serial       Atom  serial
            # number.
            line = line + str(counter).rjust(5) + ' '
            # 13 - 16        Atom          name         Atom name.
            line = line + atom_type
            # 17             Character     altLoc       Alternate
            # location indicator.
            line = line + ' '
            # 18 - 20        Residue name  resName      Residue name.
            line = line + residue_name + ' '
            # 22             Character     chainID      Chain
            # identifier.
            line = line + chain
            # 23 - 26        Integer       resSeq       Residue
            # sequence number.
            line = line + str(i + 1).rjust(4)
            # 27             AChar         iCode        Code for
            # insertion of residues.
            line = line + ' ' + '   '
            # 31 - 38        Real(8.3)     x            Orthogonal
            # coordinates for X in Angstroms.
            line = line + ('%.3f' % X[0, i]).rjust(8)
            # 39 - 46        Real(8.3)     y            Orthogonal
            # coordinates for Y in Angstroms.
            line = line + ('%.3f' % X[1, i]).rjust(8)
            # 47 - 54        Real(8.3)     z            Orthogonal
            # coordinates for Z in Angstroms.
            line = line + ('%.3f' % X[2, i]).rjust(8)
            # 55 - 60        Real(6.2)     occupancy    Occupancy.
            line = line + '  1.00'
            # 61 - 66        Real(6.2)     tempFactor   Temperature
            # factor.
            line = line + ' 12.00'
            # 77 - 78        LString(2)    element      Element symbol,
            # right-justified.
            line = line + '          ' + element
            # 79 - 80        LString(2)    charge       Charge  on the
            # atom.
            pdb_lines.append(line)
            counter = counter + 1
    return pdb_lines, counter


def coords_to_pdb_sequence(coords_sets, sequence, chain, mask=None):
    """ Write a chain to a PDB file """
    pdb_lines = ''
    counter = 1
    atom_types = [' N  ', ' CA ', ' C  ', ' O  ', ' CB ']
    elements = ['N', 'C', 'C', 'O', 'C']
    element_dict = dict(zip(atom_types, elements))
    # N CA C O SCOM
    # example = 'ATOM      1  N   HIS A 218      11.361  61.787  24.572  1.00 38.92           N  '
    pdb_lines = []
    for i in xrange(coords_sets[0].shape[1]):
        # Determine atom code
        letter = sequence[i]
        residue_name = AAcodes_inverse[letter] if letter in AAcodes_inverse else 'CSE'
        for atom_type, X in zip(atom_types, coords_sets):
            if mask is None or mask[i]:
                # N CA C O SCOM
                #  1 -  6        Record name   "ATOM  "
                line = 'ATOM  '
                # 7 - 11        Integer       serial       Atom  serial
                # number.
                line = line + str(counter).rjust(5) + ' '
                # 13 - 16        Atom          name         Atom name.
                line = line + atom_type
                # 17             Character     altLoc       Alternate
                # location indicator.
                line = line + ' '
                # 18 - 20        Residue name  resName      Residue name.
                line = line + residue_name + ' '
                # 22             Character     chainID      Chain
                # identifier.
                line = line + chain
                # 23 - 26        Integer       resSeq       Residue
                # sequence number.
                line = line + str(i + 1).rjust(4)
                # 27             AChar         iCode        Code for
                # insertion of residues.
                line = line + ' ' + '   '
                # 31 - 38        Real(8.3)     x            Orthogonal
                # coordinates for X in Angstroms.
                line = line + ('%.3f' % X[0, i]).rjust(8)
                # 39 - 46        Real(8.3)     y            Orthogonal
                # coordinates for Y in Angstroms.
                line = line + ('%.3f' % X[1, i]).rjust(8)
                # 47 - 54        Real(8.3)     z            Orthogonal
                # coordinates for Z in Angstroms.
                line = line + ('%.3f' % X[2, i]).rjust(8)
                # 55 - 60        Real(6.2)     occupancy    Occupancy.
                line = line + '  1.00'
                # 61 - 66        Real(6.2)     tempFactor   Temperature
                # factor.
                line = line + ' 12.00'
                # 77 - 78        LString(2)    element      Element symbol,
                # right-justified.
                line = line + '          ' + element_dict[atom_type]
                # 79 - 80        LString(2)    charge       Charge  on the
                # atom.
                pdb_lines.append(line)
                counter = counter + 1
    return pdb_lines

def coords_to_pdb(coords_sets):
    """ Write a chain to a PDB file """
    pdb_lines = ''
    counter = 1
    atom_types = [' N  ', ' CA ', ' C  ', ' O  ', ' CB ']
    elements = ['N', 'C', 'C', 'O', 'C']
    element_dict = dict(zip(atom_types, elements))
    # N CA C O SCOM
    # example = 'ATOM      1  N   HIS A 218      11.361  61.787  24.572  1.00 38.92           N  '
    pdb_lines = []
    for i in xrange(coords_sets[0].shape[1]):
        for atom_type, X in zip(atom_types, coords_sets):
            if np.isfinite(np.sum(X[:, i])):
                # N CA C O SCOM
                #  1 -  6        Record name   "ATOM  "
                line = 'ATOM  '
                # 7 - 11        Integer       serial       Atom  serial
                # number.
                line = line + str(counter).rjust(5) + ' '
                # 13 - 16        Atom          name         Atom name.
                line = line + atom_type
                # 17             Character     altLoc       Alternate
                # location indicator.
                line = line + ' '
                # 18 - 20        Residue name  resName      Residue name.
                line = line + 'ALA '
                # 22             Character     chainID      Chain
                # identifier.
                line = line + 'A'
                # 23 - 26        Integer       resSeq       Residue
                # sequence number.
                line = line + str(i + 1).rjust(4)
                # 27             AChar         iCode        Code for
                # insertion of residues.
                line = line + ' ' + '   '
                # 31 - 38        Real(8.3)     x            Orthogonal
                # coordinates for X in Angstroms.
                line = line + ('%.3f' % X[0, i]).rjust(8)
                # 39 - 46        Real(8.3)     y            Orthogonal
                # coordinates for Y in Angstroms.
                line = line + ('%.3f' % X[1, i]).rjust(8)
                # 47 - 54        Real(8.3)     z            Orthogonal
                # coordinates for Z in Angstroms.
                line = line + ('%.3f' % X[2, i]).rjust(8)
                # 55 - 60        Real(6.2)     occupancy    Occupancy.
                line = line + '  1.00'
                # 61 - 66        Real(6.2)     tempFactor   Temperature
                # factor.
                line = line + ' 12.00'
                # 77 - 78        LString(2)    element      Element symbol,
                # right-justified.
                line = line + '          ' + element_dict[atom_type]
                # 79 - 80        LString(2)    charge       Charge  on the
                # atom.
                pdb_lines.append(line)
                counter = counter + 1
    return pdb_lines


def extend_ABC(A, B, C, length, angle, torsion):
    # Extend a backbone using the Natural Extension Reference Frame approach
    # Given prior coordinates A,B,C place D
    # 
    # Pre-apply the rotations as though the origin were C
    angle_rads = np.radians(180 - angle)
    torsion_rads = np.radians(torsion)
    cos_angle, sin_angle = np.cos(angle_rads), np.sin(angle_rads)
    cos_torsion, sin_torsion = np.cos(torsion_rads), np.sin(torsion_rads)
    # Baseline vectors
    AB, BC = B-A, C-B
    bc = BC / np.linalg.norm(BC)
    # Normal vector to the ABC plane
    N = np.cross(AB, bc)
    N = N / np.linalg.norm(N)
    # Vector in ABC plane that is normal to BC
    NxBC = np.cross(N,BC)
    NxBC = NxBC / np.linalg.norm(NxBC)
    # Rotate into the appropriate coordinate system
    D = length * (cos_angle * bc 
                  + cos_torsion * sin_angle * NxBC 
                  + sin_torsion * sin_angle * N) + C
    return D


def build_protein(L, phi = None, psi = None):
    if phi is None:
        phi = 180 * np.ones(L)
    if psi is None:
        psi = 180 * np.ones(L)

    # Extend a backbone using the Natural Extension Reference Frame approach
    X_CA = np.zeros((3,L))
    X_C = np.zeros((3,L))
    X_N = np.zeros((3,L))
    X_O = np.zeros((3,L))
    X_SCOM = np.zeros((3,L))
    #
    # N CA C O SCOM
    # Build random initial coordinates backwards for reference frame
    C_prev =  np.zeros(3)
    CA_prev =  C_prev + np.random.randn(3)
    N_prev =  CA_prev + np.random.randn(3)
    #
    for i in xrange(L):
        omega = 180
        psi_prev = 0 if i is 0 else psi[i-1]
        N_prev = N_prev if i is 0 else X_N[:,i-1]
        CA_prev = CA_prev if i is 0 else X_CA[:,i-1]
        C_prev = C_prev if i is 0 else X_C[:,i-1]
        # Backbone atoms
        X_N[:,i] = extend_ABC(N_prev, CA_prev, C_prev, 1.32, 114, psi[i])
        X_CA[:,i] = extend_ABC(CA_prev, C_prev, X_N[:,i], 1.45, 123, omega)
        X_C[:,i] = extend_ABC(C_prev, X_N[:,i], X_CA[:,i], 1.51, 109.5, phi[i])
        # Side chains and hydrogen bonding
        X_O[:,i] = extend_ABC(X_N[:,i], X_CA[:,i], X_C[:,i], 1.24, 121, psi_prev + 180)
        X_SCOM[:,i] = extend_ABC(C_prev, X_N[:,i], X_CA[:,i], 1.51, 109.5, phi[i] - 109.5)
    # Center the coordinates
    center = np.mean(X_CA, axis=1, keepdims=True)
    X_CA = X_CA - center
    X_C = X_C - center
    X_N = X_N - center
    X_O = X_O - center
    X_SCOM = X_SCOM - center
    return [X_N, X_CA, X_C, X_O, X_SCOM]


if __name__ == "__main__":
    # Alpha helix
    L = 40
    sigma = 0
    phi = -57 * np.ones(L) + sigma * np.random.randn(L)
    psi = -47 * np.ones(L) + sigma * np.random.randn(L)

    # Antiparallel strands
    # L = 20
    # sigma = 0
    # phi = -140 * np.ones(L) + sigma * np.random.randn(L)
    # psi = 135 * np.ones(L) + sigma * np.random.randn(L)

    # Random thin
    # L = 250
    # sigma = 20
    # phi = 180 * np.ones(L) + sigma * np.random.randn(L)
    # psi = 180 * np.ones(L) + sigma * np.random.randn(L)

    # Sheet-like loopy thing
    L = 100
    sigma = 20
    phi = -115 * np.ones(L) + sigma * np.random.randn(L)
    psi = 140 * np.ones(L) + sigma * np.random.randn(L)

    pdb_lines = coords_to_pdb(build_protein(L, phi, psi))
    with open('test.pdb', 'w') as f:
        f.write('\n'.join(pdb_lines) + '\n')